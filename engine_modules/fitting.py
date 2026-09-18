"""
Model fitting and sampling utilities.
"""

from __future__ import annotations

import arviz as az
import pymc as pm

from contracts import ADVANCED_CONFIG, DEFAULT_CONFIG, FAST_CONFIG, JOINT_CONFIG, ModelConfig
from engine_modules.model import JointModelConfig


def fit_model(
    model: pm.Model,
    config: ModelConfig = DEFAULT_CONFIG,
) -> az.InferenceData:
    """
    Fit the Bayesian model with standard configuration.
    """
    with model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
    return idata


def fit_joint_model(
    model: pm.Model,
    config: JointModelConfig = JointModelConfig(),
) -> az.InferenceData:
    """
    Fit the joint SKU market-share model.
    """
    with model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
    return idata


def run_prior_predictive(
    model: pm.Model,
    draws: int = 500,
    random_seed: int = 42,
) -> az.InferenceData:
    """Run prior predictive checks."""
    with model:
        prior = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)
    return prior


def run_posterior_predictive(
    model: pm.Model,
    idata: az.InferenceData,
    draws: int = 500,
    random_seed: int = 42,
) -> az.InferenceData:
    """Run posterior predictive checks."""
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["log_velocity_std_z_obs", "sku_units_obs"],
            random_seed=random_seed,
        )
    return ppc


def get_model_config(config_name: str) -> ModelConfig:
    """Get model config by name."""
    configs = {
        "Fast (2 chains)": FAST_CONFIG,
        "Default (4 chains)": DEFAULT_CONFIG,
        "Advanced (4 chains, Student-t)": ADVANCED_CONFIG,
        "Joint SKU Share Model": JOINT_CONFIG,
    }
    return configs.get(config_name, DEFAULT_CONFIG)
