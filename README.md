# End to End advanced table detection service for any document [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![](https://img.shields.io/badge/Prateek-Ralhan-brightgreen.svg?colorB=ff0000)](https://prateekralhan.github.io/)
An end-to-end advanced table detection service for any generic document (irrespective of document type/category) using [Table Transformers by Microsoft](https://github.com/microsoft/table-transformer).

-----------------
### Please note that this is a prototype and not meant to be used directly in Production.
-----------------

The service comes in 2 packs:

* **Streamlit WebApp**:
    User Interface wherein user can upload any image which has table(s), and the app would detect and recognise the table structure (you can tune the parameters for better detection capabilities), which you can then download as CSV.

* **End to End orchestration pipeline**:
    The pipeline takes in an input feed of documents, converts their individual pages to images and then performs table detection and structure recognition on them in an asynchronous manner and all the detected tables are parsed as CSVs in the output.

  ## Flow chart:

  ![image](https://github.com/prateekralhan/document-table-detection-service/assets/29462447/de6353e1-8a2c-4a0b-800c-d43741e86b49)


## Installation:
* Simply run the command ***pip install -r requirements.txt*** to install the dependencies.

## Usage:
  * **Running the WebApp**:
      1. Clone this repository and install the dependencies as mentioned above.
      2. Navigate to `streamlit_webapp` directory and simply run the command: 
      ```
      streamlit run app.py
      ```
      3. Navigate to http://localhost:8501 in your web-browser.
      4. By default, streamlit allows us to upload files of **max. 200MB**. If you want to have more size for uploading documents, execute the command :
      ```
      streamlit run app.py --server.maxUploadSize=1028
      ```
  * **Running the pipeline**:
      1. Clone this repository and install the dependencies as mentioned above.
      2. Navigate to `pipeline` directory and simply run the command: 
      ```
      python pipeline.py
      ```



## Citations
```
@software{smock2021tabletransformer,
  author = {Smock, Brandon and Pesala, Rohith},
  month = {06},
  title = {{Table Transformer}},
  url = {https://github.com/microsoft/table-transformer},
  version = {1.0.0},
  year = {2021}
}
```
```
@inproceedings{smock2022pubtables,
  title={Pub{T}ables-1{M}: Towards comprehensive table extraction from unstructured documents},
  author={Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4634-4642},
  year={2022},
  month={June}
}
```
```
@inproceedings{smock2023grits,
  title={Gri{TS}: Grid table similarity metric for table structure recognition},
  author={Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={535--549},
  year={2023},
  organization={Springer}
}
```
```
@article{smock2023aligning,
  title={Aligning benchmark datasets for table structure recognition},
  author={Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={371--386},
  year={2023},
  organization={Springer}
}
```
