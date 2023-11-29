import boto3
from botocore.client import Config
import cx_Oracle
import numpy as np
import onnxruntime as ort
import requests
import time
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop

s3_config = {
  "s3_user": "minio",
  "s3_pass": "minio123",
  "s3_host": "http://kubeflow-minio-gateway-525eca1d5089dbdc-istio-system.apps.b2s001.pbm.ihost.com",
  "s3_bucket": "projects",
  "s3_object": "fraud-detection/model/{MODEL_VERSION}/model.onnx",
}

model_path = "model.onnx"

class HelloHandler(RequestHandler):
  def get(self):
    self.write({'message': 'hello world'})

class InferenceHandler(RequestHandler):
  def get(self):
      print("Fetching latest transactions from ORACLE...")
      cursor = self.application.db_connection.cursor()
      sql = """SELECT *
         FROM transactions
         ORDER BY "Year_Month_Day_Time" DESC"""
      cursor.execute(sql)
      transactions = cursor.fetchmany(4)
      print("Latest 4 transactions:")
      print(transactions)
      print(type(transactions))
      transactions = np.asarray(transactions).astype(np.float32)
      transactions = transactions[:,1:]
      transactions = transactions.reshape(1, 4, 103)
      print(type(transactions))
      print(transactions)

      print("Inferencing...")
      start = time.time()
      ort_inputs = {self.application.model.get_inputs()[0].name: transactions}
      preds = self.application.model.run(None, ort_inputs)[0]
      preds = np.squeeze(preds)
      end = time.time()

      self.write({'message': f"Time taken for preprocessing & inference with model v{self.application.model_version}: {end-start} s; fraud probability: {preds}"})

class UpdateModelFromS3Handler(RequestHandler):

  def initialize(self, s3_user, s3_pass, s3_host, s3_bucket, s3_object):
    self.s3_user = s3_user
    self.s3_pass = s3_pass
    self.s3_host = s3_host
    self.s3_bucket = s3_bucket
    self.s3_object = s3_object

  def get(self):
    model_version = self.get_argument("model_version", "1", True)
    s3_object = self.s3_object.replace("{MODEL_VERSION}", model_version)

    s3_client = boto3.session.Session().resource(
      service_name="s3",
      endpoint_url=self.s3_host,
      aws_access_key_id=self.s3_user,
      aws_secret_access_key=self.s3_pass,
      config=Config(signature_version="s3v4"),
    )
    bucket = s3_client.Bucket(self.s3_bucket)

    print(f"Downloading {model_path} from {self.s3_host}/{self.s3_bucket}/{s3_object}...")
    bucket.download_file(s3_object, model_path)

    print("Redeploying model...")
    del self.application.model
    self.application.model = ort.InferenceSession(model_path)
    self.application.model_version = model_version

    self.write({'message': f"Downloaded {self.s3_host}/{self.s3_bucket}/{s3_object} to {model_path} on AIX and redeployed model!"}) 

def make_app():
  urls = [
    ("/", HelloHandler),
    ("/infer", InferenceHandler),
    ("/update", UpdateModelFromS3Handler, s3_config)
  ]
  return Application(urls)
  
if __name__ == '__main__':
    app = make_app()

    print("Establishing ORACLE connection...")
    app.db_connection = cx_Oracle.connect(
      user="ADMIN",
      password="ADMIN",
      dsn="p114oracle.pbm.ihost.com:1521/kubeDB"
    )
    print("Connected to ORACLE.")

    print("Loading model...")
    app.model = ort.InferenceSession(model_path)
    app.model_version = 1
    print("Model loaded.")

    app.listen(3000)
    IOLoop.instance().start()
