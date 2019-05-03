import os
import io
import tensorflow as tf
import re

from PIL import Image
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('txt_input', '/home/stine/OIDv4_ToolKit/OID/Dataset/train/Frog/Label', 'Path to Label.txt input')
flags.DEFINE_string('output_path', '/home/stine/repositories/MSCCode/TFRecord', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '/home/stine/OIDv4_ToolKit/OID/Dataset/train/Frog', 'Path to images')
FLAGS = flags.FLAGS

def class_text_to_int(row_label):
    if row_label == 'Frog':
        return 1
    else:
        None

def create_tf_example(img_filename, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(img_filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    match = re.match(r'(.*)\.jpg', img_filename)
    filename = match.group(1)

    img_filename = img_filename.encode('utf8')
    img_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []


    label_filename = FLAGS.txt_input + '/' + filename + '.txt'
    file = open(label_filename, 'r')
    labels = file.readline()

    match = re.match(r'(.*)\s(.*)\s(.*)\s(.*)\s(.*)\s', labels)
    label = match.group(1)
    xmin = float(match.group(2))
    ymin = float(match.group(3))
    xmax = float(match.group(4))
    ymax = float(match.group(5))


    xmins.append(xmin / width)
    xmaxs.append(xmax / width)
    ymins.append(ymin / height)
    ymaxs.append(ymax / height)
    classes_text.append(label.encode('utf8'))
    classes.append(class_text_to_int(label))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(img_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    print(FLAGS.output_path)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        img_filename = os.fsdecode(file)
        tf_example = create_tf_example(img_filename, path)
        writer.write(tf_example.SerializeToString())




    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()