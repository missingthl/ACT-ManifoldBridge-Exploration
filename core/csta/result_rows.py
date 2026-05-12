from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from core.pia_audit import write_candidate_audit


AG_RESULT_FIELDS = [
    "ag_target_effective_rank",
    "ag_target_pairwise_cosine_mean",
    "ag_target_norm_mean",
    "ag_target_norm_std",
    "ag_head_pairwise_cosine_mean",
    "ag_head_effective_rank",
    "ag_head_usage_entropy",
    "ag_operator_train_mse_mean",
    "ag_operator_train_cosine_mean",
    "ag_pred_target_cosine_mean",
    "ag_tangent_available_rate",
    "ag_fallback_rate",
    "ag_pos_dist_mean",
    "ag_neg_centroid_dist_mean",
    "ag_pred_norm_mean",
    "ag_hidden_dim",
    "ag_ridge",
    "ag_activation",
    "effective_k_ag_heads",
]

CS_FLOW_RESULT_FIELDS = [
    "cs_flow_train_mse_mean",
    "cs_flow_train_cosine_mean",
    "cs_flow_pred_target_cosine_mean",
    "cs_flow_target_dist_mean",
    "cs_flow_target_dist_std",
    "cs_flow_velocity_norm_mean",
    "cs_flow_velocity_norm_std",
    "cs_flow_fallback_rate",
    "cs_flow_target_effective_rank",
    "cs_flow_target_pairwise_cosine_mean",
    "cs_flow_velocity_effective_rank",
    "cs_flow_velocity_pairwise_cosine_mean",
    "unique_direction_ratio",
    "generated_direction_pairwise_cosine_mean",
    "effective_aug_multiplier",
    "cs_flow_hidden_width",
    "cs_flow_class_embedding_dim",
    "cs_flow_hidden_layers",
    "cs_flow_t_gen",
    "cs_flow_k_same",
    "cs_flow_epochs",
    "cs_flow_batch_size",
    "cs_flow_lr",
    "cs_flow_weight_decay",
]

LATENT_RESIDUAL_RESULT_FIELDS = [
    "latent_train_cosine_loss_mean",
    "latent_train_mse_mean",
    "latent_train_pred_target_cosine_mean",
    "latent_pred_velocity_norm_mean",
    "latent_pred_velocity_norm_std",
    "latent_target_velocity_norm_mean",
    "latent_target_velocity_norm_std",
    "latent_target_dist_mean",
    "latent_target_dist_std",
    "latent_target_sampling_entropy",
    "latent_target_sampling_entropy_by_class_mean",
    "latent_target_sampling_entropy_by_class_min",
    "latent_fallback_rate",
    "latent_residual_effective_rank",
    "latent_residual_pairwise_cosine_mean",
    "latent_generated_direction_pairwise_cosine_mean",
    "latent_unique_direction_ratio",
    "latent_effective_aug_multiplier",
    "latent_pred_target_cosine_mean",
    "latent_hidden_width",
    "latent_class_embedding_dim",
    "latent_hidden_layers",
    "latent_lambda_cos",
    "latent_flow_epochs",
    "latent_flow_batch_size",
    "latent_flow_lr",
    "latent_flow_weight_decay",
    "latent_rbf_tau_floor",
]

TASK_GUIDED_LATENT_RESULT_FIELDS = [
    "task_utility_mean",
    "task_utility_std",
    "task_margin_mean",
    "task_margin_std",
    "task_bad_margin_mass",
    "task_wrong_pred_mass",
    "task_sampling_entropy",
    "task_sampling_effective_support",
    "task_kl_task_vs_geo",
    "task_guidance_fallback_rate",
    "task_guidance_fallback_reason",
    "task_invalid_candidate_rate",
    "task_warmup_train_epochs",
    "task_warmup_train_loss_mean",
    "task_guidance_beta",
    "task_guidance_margin_min",
    "task_guidance_lambda_margin",
    "task_guidance_max_candidates",
]

LC_LATENT_RESULT_FIELDS = [
    "lc_valid_candidate_rate",
    "lc_no_valid_fallback_rate",
    "lc_bad_margin_mass",
    "lc_wrong_pred_mass",
    "lc_sampling_entropy",
    "lc_sampling_effective_support",
    "lc_kl_lc_vs_geo",
    "lc_margin_mean",
    "lc_margin_std",
    "lc_margin_target_mean",
    "lc_weight_top1_mass",
    "lc_fallback_reason",
    "lc_beta",
    "lc_margin_floor",
    "lc_gamma_eps",
    "lc_warmup_epochs",
    "lc_max_candidates",
    "lc_warmup_train_loss_mean",
]

SPG_RESULT_FIELDS = [
    "spg_zhead_train_acc",
    "spg_zhead_train_loss_mean",
    "spg_grad_norm_mean",
    "spg_grad_norm_std",
    "spg_projected_grad_norm_mean",
    "spg_projected_grad_norm_std",
    "spg_projection_energy",
    "spg_projection_energy_std",
    "spg_direction_pairwise_cosine_mean",
    "spg_effective_aug_multiplier",
    "spg_support_rank",
    "spg_support_condition",
    "spg_projection_ridge",
    "spg_noise_sigma",
    "spg_zhead_epochs",
    "spg_zhead_hidden_dim",
    "ecl_projection_energy_mean",
    "ecl_projection_energy_std",
    "ecl_alpha_mean",
    "ecl_alpha_std",
    "ecl_alignment_to_projected_gradient_mean",
    "ecl_direction_pairwise_cosine_mean",
    "ecl_effective_aug_multiplier",
    "ecl_support_rank",
    "ecl_support_noise_norm_mean",
    "ecl_support_noise_norm_std",
    "ecl_fallback_rate",
    "rn_ecl_projection_energy_mean",
    "rn_ecl_projection_energy_std",
    "rn_ecl_alpha_mean",
    "rn_ecl_alpha_std",
    "rn_ecl_direction_pairwise_cosine_mean",
    "rn_ecl_alignment_to_projected_gradient_mean",
    "rn_ecl_effective_aug_multiplier",
    "rn_ecl_support_rank",
    "rn_ecl_fallback_rate",
]

GI_SPG_RESULT_FIELDS = [
    "gi_spg_operator_train_mse",
    "gi_spg_operator_train_cosine",
    "gi_spg_target_norm_mean",
    "gi_spg_target_norm_std",
    "gi_spg_pred_norm_mean",
    "gi_spg_pred_norm_std",
    "gi_spg_pred_target_cosine_mean",
    "gi_spg_projection_energy_mean",
    "gi_spg_projection_energy_std",
    "gi_spg_zhead_train_acc",
    "gi_spg_direction_pairwise_cosine_mean",
    "gi_spg_effective_aug_multiplier",
    "gi_spg_support_rank",
    "gi_spg_hidden_dim",
    "gi_spg_ridge",
    "gi_spg_activation",
]

SPG_CFM_RESULT_FIELDS = [
    "spg_cfm_train_mse_mean",
    "spg_cfm_train_cosine_mean",
    "spg_cfm_train_pred_target_cosine_mean",
    "spg_cfm_generation_pred_target_cosine_mean",
    "spg_cfm_generated_direction_pairwise_cosine_mean",
    "spg_cfm_effective_aug_multiplier",
    "spg_cfm_alignment_to_spg_mean",
    "spg_cfm_steps",
    "spg_cfm_projection_energy_mean",
    "spg_cfm_projection_energy_std",
    "spg_cfm_condition_norm_mean",
    "spg_cfm_condition_norm_std",
    "spg_zhead_train_acc",
    "gamma_used_ratio_mean",
    "transport_error_logeuc_mean",
    "augmentation_build_time_sec",
    "spg_cfm_zhead_time_sec",
    "spg_cfm_condition_time_sec",
    "spg_cfm_train_time_sec",
    "spg_cfm_generation_time_sec",
    "generation_time_per_aug_sample_ms",
]


def build_failure_result_row(*, dataset_name: str, seed: int, args, fail_reason: str) -> Dict[str, object]:
    return {
        "dataset": dataset_name,
        "seed": seed,
        "status": "failed",
        "fail_reason": str(fail_reason),
        "requested_k_dir": args.k_dir,
        "effective_k_dir": 0,
        "algo": args.algo,
        "model": args.model,
        "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
    }


def build_success_result_row(
    *,
    dataset_name: str,
    seed: int,
    args,
    pipeline_out: Dict[str, object],
    y_train: np.ndarray,
) -> Dict[str, object]:
    res_base = pipeline_out["res_base"]
    res_act = pipeline_out["res_act"]
    avg_bridge = pipeline_out["avg_bridge"]
    gain = float(res_act["macro_f1"] - res_base["macro_f1"])
    summary = {
        "dataset": dataset_name,
        "seed": seed,
        "status": "success",
        "algo": args.algo,
        "model": args.model,
        "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
        "base_f1": float(res_base["macro_f1"]),
        "act_f1": float(res_act["macro_f1"]),
        "gain": gain,
        "f1_gain_pct": gain / (float(res_base["macro_f1"]) + 1e-7) * 100.0,
        "base_stop_epoch": int(res_base.get("stop_epoch", 0)),
        "act_stop_epoch": int(res_act.get("stop_epoch", 0)),
        "base_best_val_f1": float(res_base.get("best_val_f1", 0.0)),
        "act_best_val_f1": float(res_act.get("best_val_f1", 0.0)),
        "transport_error_fro_mean": float(avg_bridge.get("transport_error_fro", 0.0)),
        "transport_error_logeuc_mean": float(avg_bridge.get("transport_error_logeuc", 0.0)),
        "bridge_cond_A_mean": float(avg_bridge.get("bridge_cond_A", 0.0)),
        "metric_preservation_error_mean": float(avg_bridge.get("metric_preservation_error", 0.0)),
        "safe_radius_ratio_mean": float(pipeline_out.get("safe_radius_ratio_mean", 1.0)),
        "manifold_margin_mean": float(pipeline_out.get("manifold_margin_mean", 0.0)),
        "gamma_requested_mean": float(pipeline_out.get("gamma_requested_mean", 0.0)),
        "gamma_used_mean": float(pipeline_out.get("gamma_used_mean", 0.0)),
        "gamma_zero_rate": float(pipeline_out.get("gamma_zero_rate", 0.0)),
        "host_geom_cosine_mean": float(pipeline_out.get("host_geom_cosine_mean", 0.0)),
        "host_conflict_rate": float(pipeline_out.get("host_conflict_rate", 0.0)),
        "candidate_total_count": int(pipeline_out.get("candidate_total_count", 0)),
        "aug_total_count": int(pipeline_out.get("aug_total_count", 0)),
        "requested_k_dir": int(args.k_dir),
        "effective_k_dir": int(pipeline_out.get("effective_k", 0)),
        "safe_clip_rate": float(pipeline_out.get("safe_clip_rate", 0.0)),
        "template_usage_entropy": float(pipeline_out.get("template_usage_entropy", 0.0)),
        "top_template_concentration": float(pipeline_out.get("top_template_concentration", 0.0)),
        "selection_stage": str(pipeline_out.get("selection_stage", "response_only")),
        "selector_name": str(pipeline_out.get("selector_name", getattr(args, "template_selection", ""))),
        "feasible_rate": float(pipeline_out.get("feasible_rate", 1.0)),
        "selector_accept_rate": float(pipeline_out.get("selector_accept_rate", 1.0)),
        "pre_filter_reject_count": int(pipeline_out.get("pre_filter_reject_count", 0)),
        "post_bridge_reject_count": int(pipeline_out.get("post_bridge_reject_count", 0)),
        "bridge_success_rate": float(pipeline_out.get("bridge_success_rate", np.nan)),
        "reject_reason_zero_gamma": int(pipeline_out.get("reject_reason_zero_gamma", 0)),
        "reject_reason_safe_radius": int(pipeline_out.get("reject_reason_safe_radius", 0)),
        "reject_reason_zero_direction": int(pipeline_out.get("reject_reason_zero_direction", 0)),
        "reject_reason_zero_margin": int(pipeline_out.get("reject_reason_zero_margin", 0)),
        "reject_reason_bridge_fail": int(pipeline_out.get("reject_reason_bridge_fail", 0)),
        "reject_reason_transport_error": int(pipeline_out.get("reject_reason_transport_error", 0)),
        "relevance_score_mean": float(pipeline_out.get("relevance_score_mean", np.nan)),
        "safe_balance_score_mean": float(pipeline_out.get("safe_balance_score_mean", np.nan)),
        "fidelity_score_mean": float(pipeline_out.get("fidelity_score_mean", np.nan)),
        "variety_score_mean": float(pipeline_out.get("variety_score_mean", np.nan)),
        "fv_score_mean": float(pipeline_out.get("fv_score_mean", np.nan)),
        "z_displacement_norm_mean": float(pipeline_out.get("z_displacement_norm_mean", 0.0)),
        "template_response_top1_mean": float(pipeline_out.get("template_response_top1_mean", np.nan)),
        "template_response_top5_mean": float(pipeline_out.get("template_response_top5_mean", np.nan)),
        "template_response_gap_top1_top5_mean": float(
            pipeline_out.get("template_response_gap_top1_top5_mean", np.nan)
        ),
        "template_response_entropy_mean": float(pipeline_out.get("template_response_entropy_mean", np.nan)),
        "pre_safe_displacement_norm_mean": float(pipeline_out.get("pre_safe_displacement_norm_mean", np.nan)),
        "post_safe_displacement_norm_mean": float(pipeline_out.get("post_safe_displacement_norm_mean", np.nan)),
        "gamma_used_ratio_mean": float(pipeline_out.get("gamma_used_ratio_mean", np.nan)),
    }
    direction_meta = dict(pipeline_out.get("direction_bank_meta", {}))
    summary["direction_bank_source"] = str(direction_meta.get("bank_source", args.algo))
    if args.algo in {"ag_target_direct", "ag_pia_single", "ag_pia_multihead5"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": "ag_pia_operator",
                "source_space": "covariance_state_operator",
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in AG_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if args.algo in {"cs_flow_target_direct", "cs_flow_single_step"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": "cs_flow_operator",
                "source_space": "covariance_state_flow",
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in CS_FLOW_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if args.algo in {"latent_residual_direct", "latent_residual_flow"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": "latent_residual_flow_operator",
                "source_space": "covariance_state_latent_residual",
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in LATENT_RESIDUAL_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if args.algo in {"task_guided_residual_direct", "task_guided_latent_residual_flow"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": "task_guided_latent_residual_operator",
                "source_space": "covariance_state_task_guided_latent_residual",
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in LATENT_RESIDUAL_RESULT_FIELDS + TASK_GUIDED_LATENT_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if args.algo in {"lc_residual_direct", "lc_latent_residual_flow"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": "label_consistent_latent_residual_operator",
                "source_space": "covariance_state_label_consistent_latent_residual",
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in LATENT_RESIDUAL_RESULT_FIELDS + LC_LATENT_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if args.algo in {
        "spg_pia_zhead",
        "spg_pia_zhead_deterministic",
        "ecl_spg_pia_zhead",
        "ecl_spg_pia_zhead_deterministic",
        "rn_ecl_spg_pia_zhead",
        "rn_ecl_spg_pia_zhead_deterministic",
    }:
        is_ecl = str(args.algo).startswith("ecl_spg_pia_")
        is_rn = str(args.algo).startswith("rn_ecl_spg_pia_")
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": (
                    "rank_normalized_ecl_spg_operator"
                    if is_rn
                    else ("energy_calibrated_langevin_spg_operator" if is_ecl else "support_projected_gradient_operator")
                ),
                "source_space": (
                    "covariance_state_rank_normalized_ecl_spg"
                    if is_rn
                    else "covariance_state_energy_calibrated_langevin_spg"
                    if is_ecl
                    else "covariance_state_support_projected_gradient"
                ),
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in SPG_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if args.algo == "gi_spg_pia_zhead":
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": "generalized_inverse_spg_operator",
                "source_space": "covariance_state_generalized_inverse_spg",
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in GI_SPG_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if args.algo in {"spg_cfm_one_step", "spg_cfm_k3"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "direction_source": "spg_conditioned_cfm_operator",
                "source_space": "covariance_state_spg_conditioned_cfm",
                "operator_source": str(args.algo),
                "template_selection": "none",
            }
        )
        for key in SPG_CFM_RESULT_FIELDS:
            if key in pipeline_out:
                summary[key] = pipeline_out[key]
    if direction_meta.get("bank_source") == "zpia_telm2" or args.algo == "zpia":
        summary.update(
            {
                "zpia_z_dim": int(direction_meta.get("z_dim", 0)),
                "zpia_n_train": int(direction_meta.get("n_train", 0)),
                "zpia_n_train_lt_z_dim": bool(direction_meta.get("n_train_lt_z_dim", False)),
                "zpia_row_norm_min": float(direction_meta.get("row_norm_min", 0.0)),
                "zpia_row_norm_max": float(direction_meta.get("row_norm_max", 0.0)),
                "zpia_row_norm_mean": float(direction_meta.get("row_norm_mean", 0.0)),
                "zpia_fallback_row_count": int(direction_meta.get("fallback_row_count", 0)),
                "telm2_recon_last": float(direction_meta.get("telm2_recon_last", 0.0)),
                "telm2_recon_mean": float(direction_meta.get("telm2_recon_mean", 0.0)),
                "telm2_recon_std": float(direction_meta.get("telm2_recon_std", 0.0)),
                "telm2_n_iters": int(direction_meta.get("telm2_n_iters", args.telm2_n_iters)),
                "telm2_c_repr": float(direction_meta.get("telm2_c_repr", args.telm2_c_repr)),
                "telm2_activation": str(direction_meta.get("telm2_activation", args.telm2_activation)),
                "telm2_bias_update_mode": str(
                    direction_meta.get("telm2_bias_update_mode", args.telm2_bias_update_mode)
                ),
            }
        )
    if args.algo in {"zpia_top1_pool", "zpia_multidir_pool", "rc4_multiz_fused"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "multi_template_pairs": int(pipeline_out.get("multi_template_pairs", 0)),
                "template_selection": str(direction_meta.get("template_selection", "anchor_top_abs_response")),
                "template_usage_entropy": float(pipeline_out.get("template_usage_entropy", 0.0)),
                "top_template_concentration": float(pipeline_out.get("top_template_concentration", 0.0)),
                "effective_k_dir_zpia": int(pipeline_out.get("effective_k_zpia", 0)),
                "u_perp_norm_ratio_mean": float(pipeline_out.get("u_perp_norm_ratio_mean", 0.0)),
                "u_perp_zero_rate": float(pipeline_out.get("u_perp_zero_rate", 0.0)),
            }
        )
        zpia_meta = dict(direction_meta.get("zpia_meta", {}))
        if zpia_meta:
            summary.update(
                {
                    "zpia_z_dim": int(zpia_meta.get("z_dim", 0)),
                    "zpia_n_train": int(zpia_meta.get("n_train", 0)),
                    "zpia_n_train_lt_z_dim": bool(zpia_meta.get("n_train_lt_z_dim", False)),
                    "zpia_row_norm_min": float(zpia_meta.get("row_norm_min", 0.0)),
                    "zpia_row_norm_max": float(zpia_meta.get("row_norm_max", 0.0)),
                    "zpia_row_norm_mean": float(zpia_meta.get("row_norm_mean", 0.0)),
                    "zpia_fallback_row_count": int(zpia_meta.get("fallback_row_count", 0)),
                    "telm2_recon_last": float(zpia_meta.get("telm2_recon_last", 0.0)),
                    "telm2_recon_mean": float(zpia_meta.get("telm2_recon_mean", 0.0)),
                    "telm2_recon_std": float(zpia_meta.get("telm2_recon_std", 0.0)),
                }
            )
        if args.algo == "rc4_multiz_fused":
            summary.update(
                {
                    "osf_alpha": float(args.osf_alpha),
                    "osf_beta": float(args.osf_beta),
                    "osf_kappa": float(args.osf_kappa),
                    "effective_k_dir_lraes": int(pipeline_out.get("effective_k_lraes", 0)),
                    "adaptive_engine_sources": ",".join([str(x) for x in direction_meta.get("engine_sources", [])]),
                    "osf_structure_overflow_rate": float(pipeline_out.get("osf_structure_overflow_rate", 0.0)),
                    "osf_alpha_eff_mean": float(pipeline_out.get("osf_alpha_eff_mean", 0.0)),
                    "osf_risk_scale_mean": float(pipeline_out.get("osf_risk_scale_mean", 0.0)),
                    "osf_risk_zero_perp_rate": float(pipeline_out.get("osf_risk_zero_perp_rate", 0.0)),
                    "osf_risk_clipped_rate": float(pipeline_out.get("osf_risk_clipped_rate", 0.0)),
                    "safe_clip_rate": float(pipeline_out.get("safe_clip_rate", 0.0)),
                    "selected_template_histogram": str(pipeline_out.get("template_counts", {})),
                }
            )
    return summary


def merge_candidate_audit_summary(
    *,
    summary: Dict[str, object],
    audit_rows: List[Dict[str, object]],
    args,
    dataset_name: str,
    seed: int,
    eta_safe,
) -> Dict[str, object]:
    if not audit_rows:
        return summary
    audit_summary = write_candidate_audit(
        audit_rows,
        out_dir=os.path.join(args.out_root, "candidate_audit"),
        dataset=dataset_name,
        seed=int(seed),
        method=str(getattr(args, "audit_method_label", "") or args.algo),
        activation_policy=str(getattr(args, "template_selection", args.algo)),
        eta_safe=eta_safe,
    )
    summary.update(audit_summary)
    return summary
