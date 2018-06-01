from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    """
    This file implements a traffic light classifier object. The object constructor loads
    a frozen TensorFlow inference graph, and the object provides a method that applies
    that inference graph to the specified image. If a traffic light is detected, the
    classifier returns its most probable state, otherwise it returns unknown.

    The classifier follows the approach described by Alex Lechner. It makes use of a 
    frozen inference graph derived by transfer learning from the pretrained SSD-2 model
    in the Google object detection API.
    """

    def __init__(self):
        """
        Creates a new classifier object and loads the frozen inference graph.
        """

        # Set probablity threshold for light detection
        self.threshold = .5

        # Set path to inference graph
        inference_graph_path = r'light_classification/frozen_inference_graph_ssd_sim.pb'

        # Create tensorflow graph object
        self.graph = tf.Graph()

        # With self.graph as the default graph...
        with self.graph.as_default():

            # 
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            # Get references to graph interface tensors
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        # reate session
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """
        Run tensorflow intference graph on the specified image and return the 
        stte of any traffic light detected in the image.

        Args:
            image: cv camera image

        Returns:
            int: traffic light color (or unknown)
        """

        # With self.graph as the default graph...
        with self.graph.as_default():

            # Fix image dimensions
            img_expand = np.expand_dims(image, axis=0)

            # Get start time
            start = datetime.datetime.now()

            # Run the inference graph
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})
            
            # Get ending time
            end = datetime.datetime.now()

            # Compute elapsed time running inference graph
            elapsed = end - start
            
            # Print elapsed time
            # print(elapsed.total_seconds())

        # Clean up outputs
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # Print classifier outputs
        print('Scores: ', scores)
        print('Classes: ', classes)

        # If the most probable class exceeds the
        # threshold...
        if scores[0] > self.threshold:

            # Convert the class number to styx traffic light
            # color enumeration and print the color...
            if classes[0] == 1:
                print('GREEN')
                return TrafficLight.GREEN

            elif classes[0] == 2:
                print('RED')
                return TrafficLight.RED

            elif classes[0] == 3:
                print('YELLOW')
                return TrafficLight.YELLOW

        # Otherwise report unknown (no traffic light or uncertain class)
        return TrafficLight.UNKNOWN


