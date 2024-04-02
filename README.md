A Flask Based Python ANPR API for GB Number Plate using Torch, Tensorflow, Tesseract OCR and yolov5 object detection.

Leverages DVLA to return information on car such as its make, model, colour, year, engine size etc.

To use simply run the app.py which will through up a Flask API, then query the API using a body key titled "image" and attached an image. 

The API will then idenitfy the registration plate and query the API from the DVLA.

You will need to attach your API-Key to an env variable called DVLA_API_KEY before hand.
