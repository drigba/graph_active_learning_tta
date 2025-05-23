from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from graph_al.evaluation.config import MetricTemplate
from graph_al.model.prediction import PredictionAttribute
from graph_al.acquisition.enum import *
from graph_al.evaluation.enum import DatasetSplit, MetricName
from graph_al.acquisition.config_tta import TTAConfig
from graph_al.test_time_adaptation.config import  AdaptationConfig
from typing import List


@dataclass
class AcquisitionStrategyConfig:
    """ Base config for acquisition strategies. """
    
    type_: AcquisitionStrategyType = MISSING
    verbose: bool = False
    num_to_acquire_per_step: int = 1 # How many nodes to acquire in each iteration
    num_steps: int = 20
    requires_model_prediction: bool = True
    balanced: bool = False # whether to select randomly but keep the class distribution balanced
    name: str | None = None
    scale: float | None = 1.0
    tta: TTAConfig | None = None
    tta_enabled: bool | None = None
    adaptation: AdaptationConfig | None = None
    adaptation_enabled: bool | None = None


    
@dataclass
class AcquisitionStrategyByAttributeConfig(AcquisitionStrategyConfig):
    """ Base config for strategies that use an attribute defined over all nodes. """
    higher_is_better: bool = MISSING
    
@dataclass
class AcquireByPredictionAttributeConfig(AcquisitionStrategyByAttributeConfig):
    """ Configuration for acquiring by querying an attriubte in the prediction. """
    
    type_: AcquisitionStrategyType = AcquisitionStrategyType.BY_PREDICTION_ATTRIBUTE
    attribute: PredictionAttribute = MISSING
    propagated: bool = MISSING
    
@dataclass
class AcquireByLogitEnergyConfig(AcquireByPredictionAttributeConfig):
    """ Configuration for acquiring by energy of softmax scores. """
    type_: AcquisitionStrategyType = AcquisitionStrategyType.LOGIT_ENERGY
    attribute: PredictionAttribute = PredictionAttribute.NONE
    higher_is_better: bool = True # High energy score (small logits) -> high uncertainty
    temperature: float = 1.0
    
@dataclass
class AcquireRandomConfig(AcquisitionStrategyConfig):
    """ Configuration for acquiring randomly. """
    
    type_: AcquisitionStrategyType = AcquisitionStrategyType.RANDOM
    requires_model_prediction: bool = False


@dataclass
class CoresetConfig(AcquisitionStrategyConfig):
    """ Uses coreset greedily to select instances. """
    
    type_: AcquisitionStrategyType = AcquisitionStrategyType.CORESET
    propagated: bool = True # Use propagated or unpropagated embeddings
    distance: CoresetDistance = MISSING
    distance_norm: float = 2
    
@dataclass
class CoresetAPPRConfig(CoresetConfig):
    
    type_: AcquisitionStrategyType = AcquisitionStrategyType.CORESET
    distance: CoresetDistance = CoresetDistance.APPR
    alpha: float = 0.2 # teleport probability
    k: int = 10 # num iterations

@dataclass
class OracleConfig(AcquisitionStrategyConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.GROUND_TRUTH
    uncertainty: OracleAcquisitionUncertaintyType = OracleAcquisitionUncertaintyType.EPISTEMIC

@dataclass
class AcquisitionStrategyFromTrainMultipleGCNsConfig(AcquisitionStrategyConfig):
    """ Configuration for acquiring based on runs of `train_multiple_gcns.py` """
    
    results_directory: str = MISSING

@dataclass
class AcquisitionStrategyBestSplitConfig(AcquisitionStrategyFromTrainMultipleGCNsConfig, AcquireRandomConfig):
    """ Configuration for setting the pool to the best split from `train_multiple_gcns.py` """
    type_: AcquisitionStrategyType = AcquisitionStrategyType.BEST_SPLIT
    metric: MetricTemplate = field(default_factory=lambda: MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.VAL))
    higher_is_better: bool = True

@dataclass
class AcquisitionStrategyBestOrderedSplitConfig(AcquisitionStrategyFromTrainMultipleGCNsConfig):
    """ Configuration for setting the pool to the best split from `optimize_best_split_order.py` """
    type_: AcquisitionStrategyType = AcquisitionStrategyType.BEST_ORDERED_SPLIT
    metric: MetricTemplate = field(default_factory=lambda: MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.VAL))
    higher_is_better: bool = True
    # Penalize "jumpy" acquisition curves
    delta_metric_penality: float = 0.01 # for each time the `metric` decreases over the run, the score is penalized with exp(delta) * delta_metric_penality
    
@dataclass
class AcquisitionStrategyFixedSequenceConfig(AcquisitionStrategyConfig):
    """ Configuration for sampling a select amount of nodes in order. """
    type_: AcquisitionStrategyType = AcquisitionStrategyType.FIXED_SEQUENCE
    order_path: str | None = None
    order: List[int] | None = None
    requires_model_prediction: bool = False
    
@dataclass
class AcquisitionStrategyByDataAttributeConfig(AcquisitionStrategyByAttributeConfig):
    """ Configuration for acquiring soley based on the input data. """
    requires_model_prediction: bool = False
    higher_is_better: bool = True
    attribute: DataAttribute = MISSING
    type_: AcquisitionStrategyType = AcquisitionStrategyType.BY_DATA_ATTRIBUTE
    

@dataclass
class AcquisitionStrategyByAPPRConfig(AcquisitionStrategyByDataAttributeConfig):
    """ Configuration for acquiring based on APPR scores of nodes. """
    attribute: DataAttribute = DataAttribute.APPR
    k: int = 10
    alpha: float = 0.2

@dataclass
class AcquisitionStrategyGEEMConfig(AcquisitionStrategyConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.GEEM
    multiprocessing: bool = False
    num_workers: int | None = None
    compute_risk_on_subset: int | None = None # On how many nodes is the risk evaluated as a subset
    subsample_pool : int | None = None # Radomly only consider a subset of the pool

@dataclass
class AcquisitionStrategyGEEMAttributeConfig(AcquisitionStrategyByAttributeConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.GEEM_ATTRIBUTE
    multiprocessing: bool = False
    num_workers: int | None = None
    compute_risk_on_subset: int | None = None # On how many nodes is the risk evaluated as a subset
    subsample_pool : int | None = None # Radomly only consider a subset of the pool
    higher_is_better: bool = False

@dataclass
class AcquisitionStrategyApproximateUncertaintyConfig(AcquisitionStrategyByAttributeConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.APPROXIMATE_UNCERTAINTY
    multiprocessing: bool = False
    num_workers: int | None = None
    subsample_pool : int | None = None # Radomly only consider a subset of the pool
    higher_is_better: bool = True 
    aleatoric_confidence_with_left_out_node: bool = False # whether to compute the aleatoric confidence of a node when leaving it out (which is more exact, but way more costly)
    aleatoric_confidence_labels_num_samples: int | None = None # How often to draw samples from the predictive distribution as truth for aleatoric confidence. If None, use the argmax label
    compute_as_ratio: bool = False # if True, it is computed as conf(alea) / conf(total), if False, we optimize the ratio of observing the remaining expected ground-truth
    features_only: bool = False # if True, only use the features for the uncertainty estimation

@dataclass
class AcquisitionStrategyLeaveOutConfig(AcquisitionStrategyByAttributeConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.LEAVE_OUT
    higher_is_better: bool = True 
    
@dataclass
class AcquisitionStrategyAugmentationRiskConfig(AcquisitionStrategyByAttributeConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.AUGMENTATION_RISK
    higher_is_better: bool = False 
    
@dataclass
class AcquisitionStrategyExpectedQueryConfig(AcquisitionStrategyByAttributeConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.EXPECTED_QUERY
    higher_is_better: bool = False 
    
@dataclass
class AcquisitionStrategyAdaptationRiskConfig(AcquisitionStrategyByAttributeConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.ADAPTATION_RISK
    higher_is_better: bool = False
    lr_feat: float = 0.0005 # learning rate for feature adaptation
    lr_adj: float = 0.1 # learning rate for structure adaptation
    epochs: int = 20 # number of epochs for feature adaptation
    strategy: str = AdaptationStrategy.DROPEDGE # strategy for augmenting the graph
    margin: float = -1 # margin for the loss function
    ratio: float = 0.1 # budget for changing the graph structure
    existing_space: bool = True # whether to enable removing edges from the graph
    loop_adj: int = 1 # number of loops for optimizing structure
    loop_feat: int = 4 # number of loops for optimizing features
    debug:int = 0 # debug flag

@dataclass
class AcquisitionStrategyAdaptationConfig(AcquisitionStrategyAdaptationRiskConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.ADAPTATION

@dataclass
class AcquisitionStrategyEducatedRandomConfig(AcquisitionStrategyAdaptationRiskConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.EDUCATED_RANDOM
    top_percent: float = 70
    low_percent: float = 1
    embedded_strategy: AcquisitionStrategyConfig = field(default_factory=AcquisitionStrategyConfig)

@dataclass
class AcquisitionStrategyTTAExpectedQueryScoreConfig(AcquisitionStrategyAdaptationRiskConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.TTA_EXPECTED_QUERY_SCORE
    embedded_strategy: AcquisitionStrategyConfig = field(default_factory=AcquisitionStrategyConfig)
    strat_node: str | None = NodeAugmentation.NOISE # which node augmentation strategy to use
    strat_edge: str | None = EdgeAugmentation.MASK # which edge augmentation strategy to use
    norm: bool | None = True
    num: int = 100 # number of tta samples
    filter: bool = False # whether to filter the tta 
    p_edge: float = 0.3 # probability of edge dropout
    p_node: float = 0.3
    higher_is_better: bool = False
    
@dataclass
class AcquisitionStrategyLatentDistanceConfig(AcquisitionStrategyByAttributeConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.LATENT_DISTANCE
    higher_is_better: bool = False 
    
@dataclass
class AcquisitionStrategyAugmentLatentConfig(AcquisitionStrategyConfig):
    """ Configuration for acquiring based on the latent space of the model. """
    type_: AcquisitionStrategyType = AcquisitionStrategyType.AUGMENT_LATENT
    num: int = 100
    p: float = 0.1
    higher_is_better: bool = False
    filter: bool = True # whether to filter the augmentations based on the prediction

@dataclass
class AcquisitionStrategyAGELikeConfig(AcquisitionStrategyConfig):
    """ Configuration for acquiring based on AGE, i.e. centrality, entropy and representativeness"""
    num_clusters: int = 6
    
@dataclass
class AcquisitionStrategyAGEConfig(AcquisitionStrategyAGELikeConfig):
    """ Configuration for acquiring based on AGE, i.e. centrality, entropy and representativeness"""
    type_: AcquisitionStrategyType = AcquisitionStrategyType.AGE
    basef: float = 0.995
    
@dataclass
class AcquisitionStrategyANRMABConfig(AcquisitionStrategyAGELikeConfig):
    """ Configuration for acquiring based on ANRMAB, i.e. centrality, entropy and representativeness combined using a multi armed bandit"""
    type_: AcquisitionStrategyType = AcquisitionStrategyType.ANRMAB
    min_probability_strategy: float = 0.2 # the minimum probability of a strategy being selected in acquisition

@dataclass
class AcquisitionStrategyFeatPropConfig(AcquisitionStrategyConfig):
    """ Configuration for acquiring based on AGE, i.e. centrality, entropy and representativeness"""
    type_: AcquisitionStrategyType = AcquisitionStrategyType.FEAT_PROP
    k: int = 2
    improved: bool = False
    add_self_loops: bool = True
    normalize: bool = True
    
@dataclass
class AcquisitionStrategySEALConfig(AcquisitionStrategyByAttributeConfig):
    """ Configuration for acquiring based on SEAL """
    type_: AcquisitionStrategyType = AcquisitionStrategyType.SEAL
    higher_is_better: bool = False # the lower the probability of being labeled, the more we want to acquire

    
@dataclass
class AcquisitionStrategyUncertaintyDifferenceConfig(AcquisitionStrategyByAttributeConfig):
    """ Configuration for computing the difference of the predictive uncertainty propagated and unpropagated. """
    type_: AcquisitionStrategyType = AcquisitionStrategyType.UNCERTAINTY_DIFFERENCE
    higher_is_better: bool = False # the lower the probability of being labeled, the more we want to acquire
    combine: str = 'ratio' # how to combine the propagated and unpropagated uncertainty
    average: str = 'prediction'

@dataclass
class AcquisitionStrategyGalaxyConfig(AcquisitionStrategyConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.GALAXY
    requires_model_prediction: bool = True
    order: int = 1
    
    
@dataclass
class AcquisitionStrategyBadgeConfig(AcquisitionStrategyConfig):
    type_: AcquisitionStrategyType = AcquisitionStrategyType.BADGE
    requires_model_prediction: bool = True

cs = ConfigStore.instance()
cs.store(name="base_config", node=AcquisitionStrategyConfig, group='acquisition_strategy')
cs.store(name="acquire_by_prediction_attribute", node=AcquireByPredictionAttributeConfig, group='acquisition_strategy')
cs.store(name="acquire_random", node=AcquireRandomConfig, group='acquisition_strategy')
cs.store(name="base_coreset", node=CoresetConfig, group='acquisition_strategy')
cs.store(name="base_coreset_appr", node=CoresetAPPRConfig, group='acquisition_strategy')
cs.store(name="base_energy", node=AcquireByLogitEnergyConfig, group='acquisition_strategy')
cs.store(name="base_best_split", node=AcquisitionStrategyBestSplitConfig, group='acquisition_strategy')
cs.store(name="base_oracle_config", node=OracleConfig, group='acquisition_strategy')
cs.store(name="base_fixed_sequence", node=AcquisitionStrategyFixedSequenceConfig, group='acquisition_strategy')
cs.store(name="base_data_attribute", node=AcquisitionStrategyByDataAttributeConfig, group='acquisition_strategy')
cs.store(name="base_appr", node=AcquisitionStrategyByAPPRConfig, group='acquisition_strategy')
cs.store(name="base_age", node=AcquisitionStrategyAGEConfig, group='acquisition_strategy')
cs.store(name="base_geem", node=AcquisitionStrategyGEEMConfig, group='acquisition_strategy')
cs.store(name="base_anrmab", node=AcquisitionStrategyANRMABConfig, group='acquisition_strategy')
cs.store(name="base_best_ordered_split", node=AcquisitionStrategyBestOrderedSplitConfig, group='acquisition_strategy')
cs.store(name="base_feat_prop", node=AcquisitionStrategyFeatPropConfig, group='acquisition_strategy')
cs.store(name="base_seal", node=AcquisitionStrategySEALConfig, group='acquisition_strategy')
cs.store(name="base_uncertainty_difference", node=AcquisitionStrategyUncertaintyDifferenceConfig, group='acquisition_strategy')
cs.store(name="base_approximate_uncertainty", node=AcquisitionStrategyApproximateUncertaintyConfig, group='acquisition_strategy')
cs.store(name="base_galaxy", node=AcquisitionStrategyGalaxyConfig, group='acquisition_strategy')
cs.store(name="base_badge", node=AcquisitionStrategyBadgeConfig, group='acquisition_strategy')
cs.store(name="base_leave_out", node=AcquisitionStrategyLeaveOutConfig, group='acquisition_strategy')
cs.store(name="base_augmentation_risk", node=AcquisitionStrategyAugmentationRiskConfig, group='acquisition_strategy')
cs.store(name="base_augment_latent", node=AcquisitionStrategyAugmentLatentConfig, group='acquisition_strategy')
cs.store(name="base_latent_distance", node=AcquisitionStrategyLatentDistanceConfig, group='acquisition_strategy')
cs.store(name="base_adaptation_risk", node=AcquisitionStrategyAdaptationRiskConfig, group='acquisition_strategy')
cs.store(name="base_adaptation", node=AcquisitionStrategyAdaptationConfig, group='acquisition_strategy')
cs.store(name="base_educated_random", node=AcquisitionStrategyEducatedRandomConfig, group='acquisition_strategy')
cs.store(name="base_expected_query", node=AcquisitionStrategyExpectedQueryConfig, group='acquisition_strategy')
cs.store(name="base_geem_attribute", node=AcquisitionStrategyGEEMAttributeConfig, group='acquisition_strategy')
cs.store(name="base_tta_expected_query_score", node=AcquisitionStrategyTTAExpectedQueryScoreConfig, group='acquisition_strategy')

# register to all initial acquisition strategy groups as well
cs.store(name="base_config", node=AcquisitionStrategyConfig, group='initial_acquisition_strategy')
cs.store(name="acquire_by_prediction_attribute", node=AcquireByPredictionAttributeConfig, group='initial_acquisition_strategy')
cs.store(name="acquire_random", node=AcquireRandomConfig, group='initial_acquisition_strategy')
cs.store(name="base_coreset", node=CoresetConfig, group='initial_acquisition_strategy')
cs.store(name="base_coreset_appr", node=CoresetAPPRConfig, group='initial_acquisition_strategy')
cs.store(name="base_energy", node=AcquireByLogitEnergyConfig, group='initial_acquisition_strategy')
cs.store(name="base_best_split", node=AcquisitionStrategyBestSplitConfig, group='initial_acquisition_strategy')
cs.store(name="base_oracle_config", node=OracleConfig, group='initial_acquisition_strategy')
cs.store(name="base_fixed_sequence", node=AcquisitionStrategyFixedSequenceConfig, group='initial_acquisition_strategy')
cs.store(name="base_data_attribute", node=AcquisitionStrategyByDataAttributeConfig, group='initial_acquisition_strategy')
cs.store(name="base_appr", node=AcquisitionStrategyByAPPRConfig, group='initial_acquisition_strategy')
cs.store(name="base_age", node=AcquisitionStrategyAGEConfig, group='initial_acquisition_strategy')
cs.store(name="base_geem", node=AcquisitionStrategyGEEMConfig, group='initial_acquisition_strategy')
cs.store(name="base_anrmab", node=AcquisitionStrategyANRMABConfig, group='initial_acquisition_strategy')
cs.store(name="base_best_ordered_split", node=AcquisitionStrategyBestOrderedSplitConfig, group='initial_acquisition_strategy')
cs.store(name="base_feat_prop", node=AcquisitionStrategyFeatPropConfig, group='initial_acquisition_strategy')
cs.store(name="base_seal", node=AcquisitionStrategySEALConfig, group='initial_acquisition_strategy')
cs.store(name="base_uncertainty_difference", node=AcquisitionStrategyUncertaintyDifferenceConfig, group='initial_acquisition_strategy')
cs.store(name="base_approximate_uncertainty", node=AcquisitionStrategyApproximateUncertaintyConfig, group='initial_acquisition_strategy')
cs.store(name="base_galaxy", node=AcquisitionStrategyGalaxyConfig, group='initial_acquisition_strategy')
cs.store(name="base_badge", node=AcquisitionStrategyBadgeConfig, group='initial_acquisition_strategy')


cs.store(name="base_geem", node=AcquisitionStrategyGEEMConfig, group='acquisition_strategy.embedded_strategy')
cs.store(name="base_geem_attribute", node=AcquisitionStrategyGEEMAttributeConfig, group='acquisition_strategy.embedded_strategy')
cs.store(name="acquire_by_prediction_attribute", node=AcquireByPredictionAttributeConfig, group='acquisition_strategy.embedded_strategy')
cs.store(name="entropy", node=AcquireByPredictionAttributeConfig, group='acquisition_strategy.embedded_strategy')
