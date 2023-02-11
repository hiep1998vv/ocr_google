# Imports the Google Cloud client library
from google.cloud import storage
import os
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "/home/adveng/ocr_google/ocr_service.json"
# os.environ["HTTPS_PROXY"]= "http://150.61.8.70:10080"
# os.environ['GRPC_DNS_RESOLVER'] = 'native'

# # Instantiates a client
# storage_client = storage.Client()

# # The name for the new bucket
# bucket_name = "my-new-bucket"

# # Creates the new bucket
# bucket = storage_client.create_bucket(bucket_name)

# print(f"Bucket {bucket.name} created.")

def authenticate_implicit_with_adc(project_id="ocr-29122022"):
    """
    When interacting with Google Cloud Client libraries, the library can auto-detect the
    credentials to use.

    // TODO(Developer):
    //  1. Before running this sample,
    //  set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc
    //  2. Replace the project variable.
    //  3. Make sure that the user account or service account that you are using
    //  has the required permissions. For this sample, you must have "storage.buckets.list".
    Args:
        project_id: The project id of your Google Cloud project.
    """

    # This snippet demonstrates how to list buckets.
    # *NOTE*: Replace the client created below with the client required for your application.
    # Note that the credentials are not specified when constructing the client.
    # Hence, the client library will look for credentials using ADC.
    storage_client = storage.Client(project=project_id)
    buckets = storage_client.list_buckets()
    print("Buckets:")
    for bucket in buckets:
        print(bucket.name)
    print("Listed all storage buckets.")



def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    # print(response.full_text_annotation.pages)
    # file = open("/home/adveng/ocr_google/result.txt", "w")
    # file.write(str(response.full_text_annotation.pages))
    # file.close()
    img_3 = cv2.imread("/home/adveng/ocr_google/test.JPG")

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            # print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                # print('Paragraph confidence: {}'.format(
                #     paragraph.confidence))

                for word in paragraph.words:
                    normal_font = 'en'
                    if (len(word.property.detected_languages)>0):
                        # print(word.property.detected_languages[0].language_code)
                        if (word.property.detected_languages[0].language_code == 'vi'):
                            normal_font = 'vi'
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    x1 = word.bounding_box.vertices[0].x
                    y1 = word.bounding_box.vertices[0].y

                    x2 = word.bounding_box.vertices[1].x
                    y2 = word.bounding_box.vertices[1].y

                    
                    if (normal_font == 'en'):
                        cv2.rectangle(img_3, (x1, y1), (x2, y2), (255,0,0), 1)
                        cv2.putText(img_3, word_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    else:
                        ## Use simsum.ttc to write Chinese.
                        fontpath = "/home/adveng/ocr_google/Be_Vietnam_Pro/BeVietnamPro-Black.ttf"     
                        font = ImageFont.truetype(fontpath, 32)
                        img_pil = Image.fromarray(img_3)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((x1, y1-10),  word_text , font = font, fill = (36,255,12))
                        img_3 = np.array(img_pil)

    cv2.imwrite("/home/adveng/ocr_google/test-edited-2nd.JPG", img_3)
    cv2.imshow('3 Channel Window', img_3)
    print("image shape: ", img_3.shape)
    cv2.waitKey()

    # cv2.destroyAllWindows()
                    # print('Word text: {} (confidence: {})'.format(
                    #     word_text, word.confidence))
                    # print("\tbox 1: {}".format(word.bounding_box.vertices[0].x))
                    
                    # print("\tbox type: {}".format(type(word.bounding_box)))
                    # for box in word.bounding_box:
                    #     print('\tBox: {}'.format(
                    #         box.vertices))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    path = "/home/adveng/ocr_google/test.JPG"
    # authenticate_implicit_with_adc()
    detect_document(path)
