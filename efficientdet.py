import json
from pathlib import Path
from typing import Any, Union, Tuple, Sequence, Optional

import tensorflow as tf

import config
from backbone import efficientnet_backbone
from bifpn import BiFPN
from cnn_layers import FilterDetections
from retinanet import RetinaNetClassifier, RetinaNetBBPredictor

TrainingOut = Tuple[tf.Tensor, tf.Tensor]
InferenceOut = Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], Sequence[tf.Tensor]]

_AVAILABLE_WEIGHTS = {None, 'imagenet', 'D0-VOC'}
_WEIGHTS_PATHS = {'D0-VOC': 'gs://ml-generic-purpose-tf-models/D0-VOC',
    # 'D0-VOC-FPN': 'gs://ml-generic-purpose-tf-models/D0-VOC-FPN'
}

class EfficientDet(tf.keras.Model):
    def __init__(self, num_classes: Optional[int] = None, D : int = 0, bidirectional: bool = True, freeze_backbone: bool = False, score_threshold: float = .1, weights : Optional[str] = 'imagenet', custom_head_classifier: bool = False, training_mode: bool = False) -> None:
        super(EfficientDet, self).__init__()

        # Check arguments coherency 
        if custom_head_classifier is True and num_classes is None:
            raise ValueError('If include_top is False, you must specify the num_classes')
        
        if weights not in _AVAILABLE_WEIGHTS:
            raise ValueError(f'Weights {weights} not available.\n'
                             f'The available weights are 
                             f'{list(_AVAILABLE_WEIGHTS)}')
        
        if ((weights is 'imagenet' or weights is None) and custom_head_classifier):
            raise ValueError('Custom Head does not make sense when training the model from scratch.
                             'Set custom_head_classifier to False or specify other weights.')

        # If weights related to efficientdet are set,
        # update the model hyperparameters according to the checkpoint, but printing a warning
        if weights != 'imagenet' and weights is not None:
            from utils.checkpoint import download_folder
            checkpoint_path = _WEIGHTS_PATHS[weights]
            save_dir = Path(download_folder(checkpoint_path))

            params = json.load((save_dir / 'hp.json').open())

            # If num_classes is specified it must be the same as in the 
            # weights checkpoint except if the custom head classifier is set to true
            if (num_classes is not None and not custom_head_classifier and num_classes != params['n_classes']):
                raise ValueError(f'Weights {weights} num classes are different from num_classes argument, please leave it as None or specify the correct classes')
            
            bidirectional = params['bidirectional']
            D = params['efficientdet']

        # Declare the model architecture
        self.config = config.EfficientDetCompudScaling(D=D)
        
        # Setup efficientnet backbone
        backbone_weights = 'imagenet' if weights == 'imagenet' else None
        self.backbone = efficientnet_backbone(self.config.B, backbone_weights)
        for l in self.backbone.layers:
            l.trainable = not freeze_backbone
        self.backbone.trainable = not freeze_backbone
        
        # Setup the feature extractor neck
        self.neck = BiFPN(self.config.Wbifpn, self.config.Dbifpn, prefix='bifpn/')

        # Setup the heads
        if num_classes is None:
            raise ValueError('You have to specify the number of classes.')

        self.num_classes = num_classes
        self.class_head = RetinaNetClassifier(self.config.Wbifpn, self.config.Dclass, num_classes=self.num_classes, prefix='class_head/')
        self.bb_head = RetinaNetBBPredictor(self.config.Wbifpn, self.config.Dclass, prefix='regress_head/')
        
        self.training_mode = training_mode

        # Inference variables, won't be used during training
        self.filter_detections = FilterDetections(config.AnchorsConfig(), score_threshold)

        # Load the weights if needed
        if weights is not None and weights != 'imagenet':
            tmp = training_mode
            self.training_mode = True
            self.build([None, *self.config.input_size, 3])
            self.load_weights(str(save_dir / 'model.h5'), by_name=True, skip_mismatch=custom_head_classifier)
            self.training_mode = tmp
            self.training_mode = tmp

            # Append a custom classifier
            if custom_head_classifier:
                self.class_head = RetinaNetClassifier(self.config.Wbifpn, self.config.Dclass, num_classes=num_classes, prefix='class_head/')

    @property
    def score_threshold(self) -> float:
        return self.filter_detections.score_threshold
    
    @score_threshold.setter
    def score_threshold(self, value: float) -> None:
        self.filter_detections.score_threshold = value

    def call(self, images: tf.Tensor, training: bool = True) -> Union[TrainingOut, InferenceOut]:
        training = training and self.training_mode
        features = self.backbone(images, training=training)
        
        # List of [BATCH, H, W, C]
        bifnp_features = self.neck(features, training=training)

        # List of [BATCH, A, 4]
        bboxes = [self.bb_head(bf, training=training) for bf in bifnp_features]

        # List of [BATCH, A, num_classes]
        class_scores = [self.class_head(bf, training=training) for bf in bifnp_features]

        # [BATCH, -1, 4]
        bboxes = tf.concat(bboxes, axis=1)

        # [BATCH, -1, num_classes]
        class_scores = tf.concat(class_scores, axis=1)

        if self.training_mode:
            return bboxes, class_scores
        else:
            return self.filter_detections(images, bboxes, class_scores)
    
    @staticmethod
    def from_pretrained(checkpoint_path: Union[Path, str], num_classes: int = None, **kwargs: Any) -> 'EfficientDet':
        from utils.checkpoint import load
        
        if (not Path(checkpoint_path).is_dir() and 
                str(checkpoint_path) not in _AVAILABLE_WEIGHTS):
            raise ValueError(f'Checkpoint {checkpoint_path} is not available')
        
        if str(checkpoint_path) in _AVAILABLE_WEIGHTS:
            checkpoint_path = _WEIGHTS_PATHS[str(checkpoint_path)]

        model, _ = load(checkpoint_path, **kwargs)

        if num_classes is not None:
            print('Loading a custom classification head...')
            model.num_classes = num_classes
            model.class_head = RetinaNetClassifier(model.config.Wbifpn, model.config.D, num_classes)

        return model