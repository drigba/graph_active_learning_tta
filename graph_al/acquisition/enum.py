from enum import unique, StrEnum

@unique
class AcquisitionStrategyType(StrEnum):
    BY_PREDICTION_ATTRIBUTE = 'by_prediction_attribute'
    LOGIT_ENERGY = 'logit_energy'
    RANDOM = 'random'
    CORESET = 'coreset'
    GROUND_TRUTH = 'ground_truth'
    BEST_SPLIT = 'best_split'
    BEST_ORDERED_SPLIT = 'best_ordered_split'
    FIXED_SEQUENCE = 'fixed_sequence'
    BY_DATA_ATTRIBUTE = 'by_data_attribute'
    AGE = 'AGE'
    GEEM = 'geem'
    ANRMAB = 'anrmab'
    FEAT_PROP = 'feat_prop'
    SEAL = 'seal'
    UNCERTAINTY_DIFFERENCE = 'uncertainty_difference'
    APPROXIMATE_UNCERTAINTY = 'approximate_uncertainty'
    GALAXY = 'galaxy'
    BADGE = 'badge'
    LEAVE_OUT = 'leave_out'
    AUGMENTATION_RISK = 'augmentation_risk'
    AUGMENT_LATENT = 'augment_latent'
    LATENT_DISTANCE = 'latent_distance'
    ADAPTATION_RISK = 'adaptation_risk'
    ADAPTATION = 'adaptation'
    EDUCATED_RANDOM = 'educated_random'
    EXPECTED_QUERY = 'expected_query'
    GEEM_ATTRIBUTE = 'geem_attribute'
    TTA_EXPECTED_QUERY_SCORE = 'tta_query_expected_score'

@unique
class NodeAugmentation(StrEnum):
    NOISE = 'noise'
    MASK = 'mask'
    ADAPTIVE = 'adaptive'
    DROPOUT = 'dropout'
    NONE = 'none'

@unique
class EdgeAugmentation(StrEnum):
    MASK = 'mask'
    ADAPTIVE = 'adaptive'
    TRAIN_CONNECTION = 'train_connection'
    NONE = 'none'
    
@unique
class AdaptationStrategy(StrEnum):
    DROPEDGE: str = 'dropedge'
    SHUFFLE: str = 'shuffle'
    DROPNODE: str = 'dropnode'
    DROPMIX: str = 'dropmix'
    DROPFEAT: str = 'dropfeat'
    FEATNOISE: str = 'featnoise'
    RWSAMPLE: str = 'rwsample'

@unique
class AdaptationIntegration(StrEnum):
    NONE: str = 'none'
    QUERY: str = 'query'
    FINE_TUNE: str = 'fine_tune'
    TRAIN: str = 'train'
    TRAIN_RECURSIVE: str = 'train_recursive'
    
@unique
class AdaptationMode(StrEnum):
    FEATURE: str = 'feature'
    STRUCTURE: str = 'structure'
    BOTH: str = 'both'
    

@unique
class CoresetDistance(StrEnum):
    
    LATENT_FEATURES = 'latent_space'
    INPUT_FEATURES = 'input_features'
    APPR = 'appr' 

@unique
class OracleAcquisitionUncertaintyType(StrEnum):
    EPISTEMIC = 'epistemic'
    ALEATORIC = 'aleatoric'
    TOTAL = 'total'
    
@unique
class DataAttribute(StrEnum):
    IN_DEGREE = 'in_degree'
    OUT_DEGREE = 'out_degree'
    APPR = 'appr'