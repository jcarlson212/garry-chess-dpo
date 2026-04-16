# src/grandmaster_dpo/eval/eval_abstractions.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, List, Optional

from grandmaster_dpo.eval.configs import SfConfig
from grandmaster_dpo.eval.single_gm.eval_sft_and_dpo_w_style_sim_utility_weight_maia2 import compute_style_score, dpo_loss_style_weighted, supervised_nll_loss
from grandmaster_dpo.eval.single_gm.eval_sft_and_dpo_w_style_v2_maia_single_gm import compute_style_score_v2
from grandmaster_dpo.eval.single_gm.eval_sft_and_dpo_w_style_v3_maia_single_gm import compute_style_score_v3, dpo_loss_style_weighted_v3
from grandmaster_dpo.eval.stockfish_eval import EvalModel
from grandmaster_dpo.eval.stockfish_helpers import make_stockfish
from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import make_config
from grandmaster_dpo.utilities.shared_style_emb_model_utils import StyleEncoder
import torch


def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)

# ============================================================
# Concrete model types
# ============================================================

class BaseMaia2(EvalModel):
    @property
    def tag(self) -> str:
        return "base_maia2"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data
    ) -> torch.Tensor:
        # Not trained; define something stable for reporting.
        # Here: mean NLL on chosen.
        return (-logp_pi_ch).mean()


class DpoModel(EvalModel):
    @property
    def tag(self) -> str:
        return "dpo"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data
    ) -> torch.Tensor:
        x = self.beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
        return -torch.nn.functional.logsigmoid(x).mean()


class SftModel(EvalModel):
    @property
    def tag(self) -> str:
        return "sft"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data
    ) -> torch.Tensor:
        # SFT objective approximated as NLL on chosen.
        return (-logp_pi_ch).mean()


class SftPairwiseModel(EvalModel):
    @property
    def tag(self) -> str:
        return "sft_pairwise"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data
    ) -> torch.Tensor:
        # Pairwise logistic loss without reference:
        # -log(sigmoid(logp(ch) - logp(rj)))
        x = (logp_pi_ch - logp_pi_rj)
        return -torch.nn.functional.logsigmoid(x).mean()
    

class SftAndDpo(EvalModel):

    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.dpo_loss_weight = dpo_loss_weight

    @property
    def tag(self) -> str:
        return f"sft_and_dpo_beta={self.beta:.2f}"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data
    ) -> torch.Tensor:
        dpo_loss = self.beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
        sft_loss = -logp_pi_ch
        return (-self.dpo_loss_weight * torch.nn.functional.logsigmoid(dpo_loss) + sft_loss).mean()

# Todo: add style loss params to constructor and heuristics for v1/v2. V3 should use actual style embedding model
class SftAndDpoWStyleV1(EvalModel):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.dpo_loss_weight = dpo_loss_weight
        self.style_tau = style_tau
        self.style_cp_scale = 40.0
        self.style_piece_bonus = 1.0
        self.style_positional_bonus = 2.0
    
    @property
    def tag(self) -> str:
        return "sft_and_dpo_w_style_v1"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data,
    ) -> torch.Tensor:
    
        style_scores = torch.tensor(
            [
                compute_style_score(
                    fen=fen,
                    chosen_uci=ch,
                    rejected_uci=rj,
                    chosen_cp=ch_cp,
                    rejected_cp=rj_cp,
                    cp_scale=self.style_cp_scale,
                    piece_bonus=self.style_piece_bonus,
                    positional_bonus=self.style_positional_bonus,
                )
                for fen, ch, rj, ch_cp, rj_cp, _, _, _, _, _ in batch_meta_data
            ],
            dtype=torch.float32,
            device=self.device,
        )

        loss = (
            self.dpo_loss_weight
            * dpo_loss_style_weighted(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=self.beta,
                tau=self.style_tau,
            )
            + supervised_nll_loss(logits_pi_m, idx_t)
        )

        return loss
    

class SftAndDpoWStyleV2(EvalModel):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.dpo_loss_weight = dpo_loss_weight
        self.style_tau = style_tau
        self.style_cp_scale = 40.0
        self.style_piece_bonus = 1.0
        self.style_positional_bonus = 2.0

    @property
    def tag(self) -> str:
        return "sft_and_dpo_w_style_v2"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data
    ) -> torch.Tensor:
        style_scores = torch.tensor(
            [
                compute_style_score_v2(
                        fen=fen,
                        chosen_uci=ch,
                        rejected_uci=rj,
                        chosen_cp=ch_cp,
                        rejected_cp=rj_cp,
                        prev_fens=prev_fens,
                        next_fens_chosen=next_fens_chosen,
                        next_fens_rejected=next_fens_rejected,
                        ply_idx=ply_idx,
                        phase=None,
                    )
                    for fen, ch, rj, ch_cp, rj_cp, ply_idx, prev_fens, next_fens_chosen, next_fens_rejected, meta_list
                    in batch_meta_data
            ],
            dtype=torch.float32,
            device=self.device,
        )

        loss = (
            self.dpo_loss_weight
            * dpo_loss_style_weighted(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=self.beta,
                tau=self.style_tau,
            )
            + supervised_nll_loss(logits_pi_m, idx_t)
        )
        return loss
    
class SftAndDpoWStyleV3(EvalModel):

    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        style_embedding_chkpt: str = "",
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.dpo_loss_weight = dpo_loss_weight
        self.style_tau = style_tau
        self.style_embedding_chkpt = style_embedding_chkpt

        style_encoder_training_cfg = make_config(
            study_name="super_v3_phi1_tau0_25_warm_from_v2final",
            train_dir="./final_experiments_for_paper/experiment2_style_model/pairs_v3_cached/train",
            eval_dir="./final_experiments_for_paper/experiment2_style_model/pairs_v2_cached/eval",
            pair_variant="v3",
            seed=42,
            embedding_dim=256,
            batch_size=4096,
            lr=3e-4,
            tau=0.25,
            phi_variant="phi1",
            epochs=8,
            max_steps_per_epoch=5000,
            max_eval_batches=150,
            num_workers=0,
            init_from_checkpoint=self.style_embedding_chkpt, # this is all that really matters for inference
            init_reset_optimizer=True,
            init_strict_load=True,
        )
        self.style_embedding_model = StyleEncoder(style_encoder_training_cfg)
        self.style_embedding_model.eval()

    @property
    def tag(self) -> str:
        return "sft_and_dpo_w_style_v3"

    @torch.inference_mode()
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data
    ) -> torch.Tensor:
        style_scores = torch.tensor(
            [
                compute_style_score_v3(
                    fen=fen,
                    chosen_uci=ch,
                    rejected_uci=rj,
                    prev_fens=prev_fens,
                    style_embedding_model=self.style_embedding_model,
                    event=meta_list["event"],
                    style_tau=self.style_tau,
                )
                for fen, ch, rj, ch_cp, rj_cp, ply_idx, prev_fens, next_fens_chosen, next_fens_rejected, meta_list
                in batch_meta_data
            ],
            dtype=torch.float32,
            device=self.device,
        )

        loss = (
            self.dpo_loss_weight
            * dpo_loss_style_weighted_v3(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=self.beta,
                tau=self.style_tau,
            )
            + supervised_nll_loss(logits_pi_m, idx_t)
        )
        return loss
    

# ------------------------------------------------------------


# Convenience “with SF helper” variants
# (These are thin wrappers that only change tag; SF is enabled by passing sf_cfg)
class DpoWithSfHelper(DpoModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None, sf_engine: Optional[Any] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"dpo_w_sf_depth_{self.depth}_pv_{self.multipv_topk}_cp_w_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"

class SftWithSfHelper(SftModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None, sf_engine: Optional[Any] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"{super().tag}_d_{self.depth}_pv_{self.multipv_topk}_cp_w_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"    
    
class SftPairwiseWithSfHelper(SftPairwiseModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None, sf_engine: Optional[Any] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"{super().tag}_d_{self.depth}_pv_{self.multipv_topk}_cp_w_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"    
    
class SftAndDpoWithSfHelper(SftAndDpo):

    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window

    @property
    def tag(self) -> str:
        return f"{super().tag}_d_{self.depth}_pv_{self.multipv_topk}_cp_w_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"    
    
class SftAndDpoWStyleV1WithSfHelper(SftAndDpoWStyleV1):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, style_tau=style_tau, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window

    @property
    def tag(self) -> str:
        return f"{super().tag}_d_{self.depth}_pv_{self.multipv_topk}_cp_w_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"    
    
class SftAndDpoWStyleV2WithSfHelper(SftAndDpoWStyleV2):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, style_tau=style_tau, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window

    @property
    def tag(self) -> str:
        return f"{super().tag}_d_{self.depth}_pv_{self.multipv_topk}_cp_w_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"    
    
class SftAndDpoWStyleV3WithSfHelper(SftAndDpoWStyleV3):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        embedding_model_chkpt_name: str = "",
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, style_tau=style_tau, style_embedding_chkpt=embedding_model_chkpt_name, sf_cfg=sf_cfg, sf_engine=sf_engine)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        self.style_embedding_chkpt = embedding_model_chkpt_name
        print(f"Style embedding checkpoint filename is: {Path(embedding_model_chkpt_name).parent.name}")

    @property
    def tag(self) -> str:
        return f"{super().tag}_d_{self.depth}_pv_{self.multipv_topk}_cp_w_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}_emb_chkpt_{Path(self.style_embedding_chkpt).parent.name.replace('__bs-4096_', '').replace('__seed-42', '')}"


# ============================================================
# Factory: instantiate the family you want with shared args
# ============================================================

def build_models_for_gm(
    *,
    maia_type: str,
    device: torch.device,
    gm_dir: Path,
    sf_cfgs: Optional[List[SfConfig]],
    beta: float,
    disable_initial_model_types: bool = False,
) -> List[EvalModel]:
    """
    gm_dir expected to contain:
      - policy_dpo_best.pt
      - policy_sft_best.pt
      - policy_pairwise_sft_best.pt
    Adjust filenames as needed.
    """
    dpo_pt = gm_dir / "policy_dpo_best.pt"
    sft_pt = gm_dir / "policy_sft_best.pt"
    pw_pt = gm_dir / "policy_pairwise_sft_best.pt"

    

    # base only
    models.append(BaseMaia2(maia_type=maia_type, device=device, policy_pt_path=None, beta=beta, sf_cfg=None, sf_engine=None))
    if not disable_initial_model_types:
        # non-SF runs
        models.append(DpoModel(maia_type=maia_type, device=device, policy_pt_path=str(dpo_pt), beta=beta, sf_cfg=None, sf_engine=None))
        models.append(SftModel(maia_type=maia_type, device=device, policy_pt_path=str(sft_pt), beta=beta, sf_cfg=None, sf_engine=None))
        models.append(SftPairwiseModel(maia_type=maia_type, device=device, policy_pt_path=str(pw_pt), beta=beta, sf_cfg=None, sf_engine=None))

    # SF-helper runs (depth lives in sf_cfg.depth)
    if sf_cfgs is not None:
        for sf_cfg in sf_cfgs:
            sf_engine = make_stockfish(
                sf_cfg.stockfish_path,
                threads=int(sf_cfg.threads),
                hash_mb=int(sf_cfg.hash_mb),
                uci_elo=sf_cfg.uci_elo,
                skill_level=None,
                timeout=float(sf_cfg.timeout_s),
            )
            models.append(DpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=str(dpo_pt), beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine))
            models.append(SftWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=str(sft_pt), beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine))
            models.append(SftPairwiseWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=str(pw_pt), beta=beta, sf_cfg=sf_cfg, sf_engine=sf_engine))
            beta = 0.6
            dpo_loss_weight = 0.1
            style_tau = 0.25
            models.append(SftAndDpoWithSfHelper(maia_type=maia_type, 
                                                device=device, 
                                                policy_pt_path=f"{gm_dir}/policy_best_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}.pt", 
                                                beta=beta, 
                                                dpo_loss_weight=dpo_loss_weight,
                                                sf_cfg=sf_cfg, 
                                                sf_engine=sf_engine))
            
            models.append(SftAndDpoWStyleV1WithSfHelper(maia_type=maia_type, 
                                                        device=device, 
                                                        policy_pt_path=f"{gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                        beta=beta, 
                                                        dpo_loss_weight=dpo_loss_weight, 
                                                        style_tau=style_tau, 
                                                        sf_cfg=sf_cfg, 
                                                        sf_engine=sf_engine))
            
            models.append(SftAndDpoWStyleV2WithSfHelper(maia_type=maia_type, 
                                                        device=device, 
                                                        policy_pt_path=f"{gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                        beta=beta, 
                                                        dpo_loss_weight=dpo_loss_weight,
                                                        style_tau=style_tau,
                                                        sf_cfg=sf_cfg, 
                                                        sf_engine=sf_engine))

    return models


def build_sf_models_for_gm(
    *,
    maia_type: str,
    device: torch.device,
    experiment1_gm_dir: Path = Path("./final_experiments_for_paper/experiment1/trained_models_twic/carlsen/"),
    experiment2_gm_dir: Path = Path("./final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic/carlsen/"),
    style_embedding_model_dir: Path = Path('./final_experiments_for_paper/experiments2_style_model/trained_models/'),
    sf_cfgs: List[SfConfig],
) -> Generator[EvalModel, None, None]:
    """
    gm_dir expected to contain:
      - policy_dpo_best.pt
      - policy_sft_best.pt
      - policy_pairwise_sft_best.pt
    Adjust filenames as needed.
    """
    dpo_pt = experiment1_gm_dir / "policy_dpo_best.pt"
    sft_pt = experiment1_gm_dir / "policy_sft_best.pt"
    pw_pt = experiment1_gm_dir / "policy_pairwise_sft_best.pt"
    sft_and_dpo_pt = experiment1_gm_dir / "policy_sft_and_dpo_best.pt"
    sft_and_dpo_w_style_v1_pt = experiment1_gm_dir / "policy_sft_and_dpo_w_style_v1_best.pt"
    sft_and_dpo_w_style_v2_pt = experiment1_gm_dir / "policy_sft_and_dpo_w_style_v2_best.pt"

    sft_and_dpo_w_style_v3_pt = experiment2_gm_dir / "policy_sft_and_dpo_w_style_v3_best.pt"

    final_v2_embedding_model_name = "final_v2_phi1_tau0_25_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42"
    final_v3_embedding_model_name = "final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42"

    models: List[EvalModel] = []

    # SF-helper runs (depth lives in sf_cfg.depth)
    for sf_cfg in sf_cfgs:
        sf_engine = make_stockfish(
            sf_cfg.stockfish_path,
            threads=int(sf_cfg.threads),
            hash_mb=int(sf_cfg.hash_mb),
            uci_elo=sf_cfg.uci_elo,
            skill_level=None,
            timeout=float(sf_cfg.timeout_s),
        )
        yield DpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_dpo_beta=0.60.pt", beta=0.6, sf_cfg=sf_cfg, sf_engine=sf_engine)
        yield SftWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_sft_best.pt", beta=0.6, sf_cfg=sf_cfg, sf_engine=sf_engine)
        yield SftPairwiseWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_pairwise_sft_best.pt", beta=0.6, sf_cfg=sf_cfg, sf_engine=sf_engine)

        beta = 0.6
        dpo_loss_weight = 0.1
        style_tau = 0.25
        for dpo_loss_weight in [0.2, 0.4]:
            yield SftAndDpoWithSfHelper(maia_type=maia_type, 
                                                device=device, 
                                                policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}.pt", 
                                                beta=beta, 
                                                dpo_loss_weight=dpo_loss_weight,
                                                sf_cfg=sf_cfg,
                                                sf_engine=sf_engine)       
            #yield SftAndDpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_beta=0.60_dpo_loss_weight=0.20.pt", beta=0.6, dpo_loss_weight=0.2, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_beta=0.60_dpo_loss_weight=0.40.pt", beta=0.6, dpo_loss_weight=0.4, sf_cfg=sf_cfg)

        for dpo_loss_weight in [0.1, 0.2]:
            for style_tau in [0.25, 0.75, 1.25]:
                yield SftAndDpoWStyleV1WithSfHelper(maia_type=maia_type, 
                                                            device=device, 
                                                            policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                            beta=beta, 
                                                            dpo_loss_weight=dpo_loss_weight, 
                                                            style_tau=style_tau, 
                                                            sf_cfg=sf_cfg,
                                                            sf_engine=sf_engine)    
            #yield SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=0.75, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=1.25, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.25, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.75, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=1.25, sf_cfg=sf_cfg)

                yield SftAndDpoWStyleV2WithSfHelper(maia_type=maia_type, 
                                                            device=device, 
                                                            policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                            beta=beta, 
                                                            dpo_loss_weight=dpo_loss_weight,
                                                            style_tau=style_tau,
                                                            sf_cfg=sf_cfg,
                                                            sf_engine=sf_engine)    
            #yield SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=0.75, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=1.25, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.25, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.75, sf_cfg=sf_cfg)
            #yield SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=1.25, sf_cfg=sf_cfg)

        for dpo_loss_weight in [0.6, 0.4, 0.8, 0.2, 1.0, 0.1]:
            yield SftAndDpoWStyleV3WithSfHelper(maia_type=maia_type, 
                                                        device=device, 
                                                        policy_pt_path=f"{experiment2_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_tau={style_tau:.2f}_embedding_model=final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42.pt", 
                                                        beta=beta, 
                                                        dpo_loss_weight=dpo_loss_weight,
                                                        style_tau=style_tau,
                                                        embedding_model_chkpt_name=f"{style_embedding_model_dir}/final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42/best.pt",
                                                        sf_cfg=sf_cfg,
                                                        sf_engine=sf_engine)   
        #yield SftAndDpoWithSfHelperWStyleV3(maia_type=maia_type, device=device, policy_pt_path=f"{experiment2_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight=0.10_style_tau=0.25_embedding_model={final_v3_embedding_model_name}.pt", beta=0.6, dpo_loss_weight=0.1, style_tau_inference=0.25, embedding_model_name=final_v3_embedding_model_name, sf_cfg=sf_cfg)

        for dpo_loss_weight in [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
            for style_tau in [0.25, 0.75, 1.25]:
                #yield SftAndDpoWithSfHelperWStyleV3(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight={dpo_loss_weight:.2f}_style_tau={style_tau:.2f}_embedding_model={final_v2_embedding_model_name}.pt", beta=0.6, dpo_loss_weight=dpo_loss_weight, style_tau_inference=style_tau, embedding_model_name=final_v2_embedding_model_name, sf_cfg=sf_cfg)
                #yield SftAndDpoWithSfHelperWStyleV3(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight={dpo_loss_weight:.2f}_style_tau={style_tau:.2f}_embedding_model={final_v3_embedding_model_name}.pt", beta=0.6, dpo_loss_weight=dpo_loss_weight, style_tau_inference=style_tau, embedding_model_name=final_v3_embedding_model_name, sf_cfg=sf_cfg)
                pass

