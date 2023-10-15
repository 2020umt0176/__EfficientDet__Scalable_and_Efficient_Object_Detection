import click
#import click library for CLI
from typing import Any

import tensorflow as tf

import matplotlib.pyplot as plt

import efficientdet
# for importing the model


@click.command()
@click.option('--image', type=click.Path(dir_okay=False, exists=True))
@click.option('--checkpoint', type=click.Path())
@click.option('--score', type=float, default=.4)

@click.option('--format', type=click.Choice(['VOC', 'labelme']),
              required=True, help='Dataset to use for training')

def main(**kwargs: Any) -> None:

    model, params = efficientdet.checkpoint.load(
        kwargs['checkpoint'], score_threshold=kwargs['score'])
        
    if kwargs['format'] == 'labelme':
        classes = params['classes_names'].split(',') 
      #splitting class names split by a comma

    elif kwargs['format'] == 'VOC':
        classes = efficientdet.data.voc.IDX_2_LABEL
    
    # load image
    im_size = model.config.input_size #input model size
    im = efficientdet.utils.io.load_image(kwargs['image'], im_size)
    norm_image = efficientdet.data.preprocess.normalize_image(im)#normalization of the image is performed
    boxes, labels, scores = model(tf.expand_dims(norm_image, axis=0), 
                                  training=False)

    labels = [classes[l] for l in labels[0]]
    scores = scores[0]
    im = efficientdet.visualizer.draw_boxes(
        im, boxes[0], labels=labels, scores=scores)
    #draw bounding boxes for the respective images
    #display these images with the bounding boxes using matplotlib
    plt.imshow(im)
    plt.axis('off')
    plt.show(block=True)

if __name__ == "__main__":
    main()
