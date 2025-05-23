from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import (
    AcquisitionStrategyBestOrderedSplitConfig, AcquisitionStrategyConfig, AcquireByPredictionAttributeConfig, 
    AcquireByLogitEnergyConfig, CoresetConfig, AcquireRandomConfig,
    CoresetAPPRConfig, CoresetDistance, AcquisitionStrategyBestSplitConfig,
    OracleConfig, AcquisitionStrategyFixedSequenceConfig, AcquisitionStrategyByDataAttributeConfig, 
    DataAttribute, AcquisitionStrategyAGEConfig,
    AcquisitionStrategyGEEMConfig, AcquisitionStrategyANRMABConfig,
    AcquisitionStrategyFeatPropConfig, AcquisitionStrategySEALConfig,
    AcquisitionStrategyUncertaintyDifferenceConfig,
    AcquisitionStrategyApproximateUncertaintyConfig,
    AcquisitionStrategyGalaxyConfig,
    AcquisitionStrategyBadgeConfig,
    AcquisitionStrategyLeaveOutConfig,
    AcquisitionStrategyAugmentationRiskConfig,
    AcquisitionStrategyAugmentLatentConfig,
    AcquisitionStrategyLatentDistanceConfig,
    AcquisitionStrategyAdaptationRiskConfig,
    AcquisitionStrategyAdaptationConfig,
    AcquisitionStrategyEducatedRandomConfig,
    AcquisitionStrategyExpectedQueryConfig,
    AcquisitionStrategyGEEMAttributeConfig,
    AcquisitionStrategyTTAExpectedQueryScoreConfig
    
)
from graph_al.acquisition.prediction_attribute import AcquisitionStrategyByPredictionAttribute
from graph_al.acquisition.random import AcquisitionStrategyRandom
from graph_al.acquisition.coreset import AcquisitionStrategyCoreset, AcquisitionStrategyCoresetAPPR
from graph_al.acquisition.energy import AcquisitionStrategyByLogitEnergy
from graph_al.acquisition.best_split import AcquisitionStrategyBestSplit, AcquisitionStrategyBestOrderedSplit
from graph_al.acquisition.oracle import AcquisitionStrategyOracle
from graph_al.acquisition.fixed_sequence import AcquisitionStrategyFixedSequence
from graph_al.acquisition.data_attribute import AcquisitionStrategyByDataAttribute, AcquisitionStrategyByAPPR
from graph_al.acquisition.age import AcquisitionStrategyAGE
from graph_al.acquisition.geem import AcquisitionStrategyGraphExpectedErrorMinimization
from graph_al.acquisition.anrmab import AcquisitionStrategyANRMAB
from graph_al.acquisition.feat_prop import AcquisitonStrategyFeatProp
from graph_al.acquisition.seal import AcquisitionStrategySEAL
from graph_al.acquisition.uncertainty_difference import AcquisitionStrategyUncertaintyDifference
from graph_al.acquisition.approximate_uncertainty import AcquisitionStrategyApproximateUncertainty
from graph_al.acquisition.badge import AcquisitionStrategyBadge
from graph_al.model.base import BaseModel
from graph_al.data.base import Dataset
from graph_al.acquisition.galaxy import AcquisitionStrategyGalaxy
from graph_al.acquisition.leave_out import AcquisitionStrategyLeaveOut
from graph_al.acquisition.augmentation_risk import AcquisitionStrategyAugmentationRisk
from graph_al.acquisition.expected_query import AcquisitionStrategyExpectedQuery
from graph_al.acquisition.augment_latent import AcquisitionStrategyAugmentLatent
from graph_al.acquisition.latent_distance import AcquisitionStrategyLatentDistance
from graph_al.acquisition.adaptation_risk import AcquisitionStrategyAdaptationRisk
from graph_al.acquisition.adaptation import AcquisitionStrategyAdaptation
from graph_al.acquisition.educated_random import AcquisitionStrategyEducatedRandom
from graph_al.acquisition.geem_attribute import AcquisitionStrategyGEEMAttribute
from graph_al.acquisition.tta_expected_score import AcquisitionStrategyTTAExpectedQueryScore

import torch

def get_acquisition_strategy(config: AcquisitionStrategyConfig, dataset: Dataset) -> BaseAcquisitionStrategy:
    match config.type_:
        case AcquireByPredictionAttributeConfig.type_:
            return AcquisitionStrategyByPredictionAttribute(config) # type: ignore
        case AcquireRandomConfig.type_:
            return AcquisitionStrategyRandom(config) # type: ignore
        case CoresetConfig.type_:
            if config.distance == CoresetDistance.APPR: # type: ignore
                return AcquisitionStrategyCoresetAPPR(config) # type: ignore
            else:
                return AcquisitionStrategyCoreset(config) # type: ignore
        case AcquireByLogitEnergyConfig.type_:
            return AcquisitionStrategyByLogitEnergy(config) # type: ignore
        case AcquisitionStrategyBestSplitConfig.type_:
            return AcquisitionStrategyBestSplit(config) # type: ignore
        case OracleConfig.type_: 
            return AcquisitionStrategyOracle(config) # type: ignore
        case AcquisitionStrategyFixedSequenceConfig.type_:
            return AcquisitionStrategyFixedSequence(config, num_nodes=dataset.num_nodes) # type: ignore
        case AcquisitionStrategyByDataAttributeConfig.type_:
            if config.attribute == DataAttribute.APPR: # type: ignore
                return AcquisitionStrategyByAPPR(config) # type: ignore
            else:
                return AcquisitionStrategyByDataAttribute(config) # type: ignore
        case AcquisitionStrategyAGEConfig.type_:
            return AcquisitionStrategyAGE(config) # type: ignore
        case AcquisitionStrategyGEEMConfig.type_:
            return AcquisitionStrategyGraphExpectedErrorMinimization(config) # type: ignore
        case AcquisitionStrategyANRMABConfig.type_:
            return AcquisitionStrategyANRMAB(config, dataset.num_nodes) # type: ignore
        case AcquisitionStrategyBestOrderedSplitConfig.type_:
            return AcquisitionStrategyBestOrderedSplit(config, num_nodes=dataset.num_nodes) # type: ignore
        case AcquisitionStrategyFeatPropConfig.type_:
            return AcquisitonStrategyFeatProp(config) # type: ignore
        case AcquisitionStrategySEALConfig.type_:
            return AcquisitionStrategySEAL(config) # type: ignore
        case AcquisitionStrategyUncertaintyDifferenceConfig.type_:
            return AcquisitionStrategyUncertaintyDifference(config) # type: ignore
        case AcquisitionStrategyApproximateUncertaintyConfig.type_:
            return AcquisitionStrategyApproximateUncertainty(config) # type: ignore
        case AcquisitionStrategyGalaxyConfig.type_:
            return AcquisitionStrategyGalaxy(config) # type: ignore
        case AcquisitionStrategyBadgeConfig.type_:
            return AcquisitionStrategyBadge(config) # type: ignore
        case AcquisitionStrategyLeaveOutConfig.type_:
            return AcquisitionStrategyLeaveOut(config) # type: ignore
        case AcquisitionStrategyAugmentationRiskConfig.type_:
            return AcquisitionStrategyAugmentationRisk(config) # type: ignore
        case AcquisitionStrategyAugmentLatentConfig.type_:
            return AcquisitionStrategyAugmentLatent(config) # type: ignore
        case AcquisitionStrategyLatentDistanceConfig.type_:
            return AcquisitionStrategyLatentDistance(config) # type: ignore
        case AcquisitionStrategyAdaptationRiskConfig.type_:
            return AcquisitionStrategyAdaptationRisk(config) # type: ignore
        case AcquisitionStrategyAdaptationConfig.type_:
            return AcquisitionStrategyAdaptation(config) # type: ignore
        case AcquisitionStrategyEducatedRandomConfig.type_:
            return AcquisitionStrategyEducatedRandom(config) # type: ignore
        case AcquisitionStrategyExpectedQueryConfig.type_:
            return AcquisitionStrategyExpectedQuery(config) # type: ignore
        case AcquisitionStrategyGEEMAttributeConfig.type_:
            return AcquisitionStrategyGEEMAttribute(config)
        case AcquisitionStrategyTTAExpectedQueryScoreConfig.type_:
            return AcquisitionStrategyTTAExpectedQueryScore(config)
        case _:
            raise ValueError(f'Unsupported acquisition strategy {config.type_}')