import re
import os
import cv2
import time
import json
import torch
import string
import random
import urllib3
import asyncio
import warnings
import tempfile
import numpy as np
import pandas as pd
from config import *
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
from collections import Counter
from itertools import count, tee
from pdf2image import convert_from_path
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

timestr=time.strftime("%Y.%m.%d-%H.%M.%S")
#ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
ocr = PaddleOCR(use_angle_cls=True, lang="german", use_gpu=False)

table_detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
table_recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

###########################################
# Image Processing Functions' definitions #
###########################################

def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


async def pytess(cell_pil_img, threshold: float = 0.5):
    cell_pil_img = TableExtractionPipeline.add_padding(
        pil_img=cell_pil_img,
        top=50,
        right=30,
        bottom=50,
        left=30,
        color=(255, 255, 255),
    )
    
    
    #cell_pil_img.save(BytesIO(), format='PNG')
    # Save the image to a temporary file
    #with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
    #    temp_file_path = temp_file.name
    #    cell_pil_img.save(temp_file_path)
    #result = ocr.process_local_file(temp_file_path).decode('utf-8')
    
    result = ocr.ocr(np.asarray(cell_pil_img), cls=True)[0]
    text = ""
    if result != None:
        txts = [line[1][0] for line in result]
        text = " ".join(txts)
    return text


def sharpen_image(pil_img):
    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img


def uniquify(seq, suffs=count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).
    Credit: https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list
    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k, v in Counter(seq).items() if v > 1]

    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix

    return seq


def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

    result = cv2.GaussianBlur(thresh, (5, 5), 0)
    result = 255 - result
    return cv_to_PIL(result)


def td_postprocess(pil_img):
    """
    Removes gray background from tables
    """
    img = PIL_to_cv(pil_img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))  # (0, 0, 100), (255, 5, 255)
    nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))  # (0, 0, 5), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3, 3)))  # (3,3)
    mask = mask & nzmask

    new_img = img.copy()
    new_img[np.where(mask)] = 255

    return cv_to_PIL(new_img)

###########################################
###########################################

def table_detector(image, THRESHOLD_PROBA):
    """
    Table detection using DEtect-object TRansformer pre-trained on 1 million tables

    """

    feature_extractor = DetrImageProcessor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = table_detection_model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]["boxes"][keep]

    return (probas[keep], bboxes_scaled)


def table_struct_recog(image, THRESHOLD_PROBA):
    """
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    """

    feature_extractor = DetrImageProcessor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = table_recognition_model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]["boxes"][keep]

    return (probas[keep], bboxes_scaled)

class TableExtractionPipeline:
    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    # colors = ["red", "blue", "green", "red", "red", "red"]

    @staticmethod
    def add_padding(pil_img, top, right, bottom, left, color=(255, 255, 255)):
        """
        Image padding as part of TSR pre-processing to prevent missing table edges
        """
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def plot_results_detection(
        self,
        model,
        pil_img,
        prob,
        boxes,
        delta_xmin,
        delta_ymin,
        delta_xmax,
        delta_ymax,
    ):
        """
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
        """
        plt.imshow(pil_img)
        ax = plt.gca()

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            xmin, ymin, xmax, ymax = (
                xmin - delta_xmin,
                ymin - delta_ymin,
                xmax + delta_xmax,
                ymax + delta_ymax,
            )
            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    color="red",
                    linewidth=3,
                )
            )
            text = f"{model.config.id2label[cl.item()]}: {p[cl]:0.2f}"
            ax.text(
                xmin - 20,
                ymin - 50,
                text,
                fontsize=10,
                bbox=dict(facecolor="yellow", alpha=0.5),
            )
        plt.axis("off")

    def crop_tables(
        self, image_path, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax
    ):
        """
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
        """
        cropped_img_list = []
        ctr = 0
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            xmin, ymin, xmax, ymax = (
                xmin - delta_xmin,
                ymin - delta_ymin,
                xmax + delta_xmax,
                ymax + delta_ymax,
            )
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))    
            cropped_img_list.append(cropped_img)
        

            img= cv2.imread(image_path)
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            try:
                cropped_image = img[ymin:ymax, xmin:xmax]
                ctr += 1
                cropped_img_path = re.sub("((.png)|(.jp(e)?g)|(.bmp)|(page_level_images\/))","", image_path)
                cropped_img_name = f"{cropped_img_path}_cropped_img_{ctr}.png"
                cv2.imwrite(f"{cropped_img_dir}/{cropped_img_name}", cropped_image) 
            except TypeError:
                continue
        return cropped_img_list

    def generate_structure(
        self,
        model,
        pil_img,
        prob,
        boxes,
        expand_rowcol_bbox_top,
        expand_rowcol_bbox_bottom,
    ):
        """
        Co-ordinates are adjusted here by 3 'pixels'
        To plot table pillow image and the TSR bounding boxes on the table
        """
        plt.figure(figsize=(32, 20))
        plt.imshow(pil_img)
        ax = plt.gca()
        rows = {}
        cols = {}
        idx = 0

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]
            text = f"{class_text}: {p[cl]:0.2f}"
            # or (class_text == 'table column')
            if (
                (class_text == "table row")
                or (class_text == "table projected row header")
                or (class_text == "table column")
            ):
                ax.add_patch(
                    plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        color=self.colors[cl.item()],
                        linewidth=2,
                    )
                )
                ax.text(
                    xmin - 10,
                    ymin - 10,
                    text,
                    fontsize=5,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

            if class_text == "table row":
                rows["table row." + str(idx)] = (
                    xmin,
                    ymin - expand_rowcol_bbox_top,
                    xmax,
                    ymax + expand_rowcol_bbox_bottom,
                )
            if class_text == "table column":
                cols["table column." + str(idx)] = (
                    xmin,
                    ymin - expand_rowcol_bbox_top,
                    xmax,
                    ymax + expand_rowcol_bbox_bottom,
                )

            idx += 1

        plt.axis("on")
        return rows, cols

    def sort_table_featuresv2(self, rows: dict, cols: dict):
        # Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
        rows_ = {
            table_feature: (xmin, ymin, xmax, ymax)
            for table_feature, (xmin, ymin, xmax, ymax) in sorted(
                rows.items(), key=lambda tup: tup[1][1]
            )
        }
        cols_ = {
            table_feature: (xmin, ymin, xmax, ymax)
            for table_feature, (xmin, ymin, xmax, ymax) in sorted(
                cols.items(), key=lambda tup: tup[1][0]
            )
        }

        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows: dict, cols: dict):
        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img

        return rows, cols

    def object_to_cellsv2(
        self,
        master_row: dict,
        cols: dict,
        expand_rowcol_bbox_top,
        expand_rowcol_bbox_bottom,
        padd_left,
    ):
        """Removes redundant bbox for rows&columns and divides each row into cells from columns
        Args:

        Returns:


        """
        cells_img = {}
        header_idx = 0
        row_idx = 0
        previous_xmax_col = 0
        new_cols = {}
        new_master_row = {}
        previous_ymin_row = 0
        new_cols = cols
        new_master_row = master_row
        ## Below 2 for loops remove redundant bounding boxes ###
        # for k_col, v_col in cols.items():
        #     xmin_col, _, xmax_col, _, col_img = v_col
        #     if (np.isclose(previous_xmax_col, xmax_col, atol=5)) or (xmin_col >= xmax_col):
        #         print('Found a column with double bbox')
        #         continue
        #     previous_xmax_col = xmax_col
        #     new_cols[k_col] = v_col

        # for k_row, v_row in master_row.items():
        #     _, ymin_row, _, ymax_row, row_img = v_row
        #     if (np.isclose(previous_ymin_row, ymin_row, atol=5)) or (ymin_row >= ymax_row):
        #         print('Found a row with double bbox')
        #         continue
        #     previous_ymin_row = ymin_row
        #     new_master_row[k_row] = v_row
        ######################################################
        for k_row, v_row in new_master_row.items():
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols) - 1:
                    xb = xmax
                xa, ya, xb, yb = xa, ya, xb, yb

                try:
                    row_img_cropped = row_img.crop((xa, ya, xb, yb))
                    row_img_list.append(row_img_cropped)
                except ValueError:
                    continue

            cells_img[k_row + "." + str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row) - 1

    def clean_dataframe(self, df):
        """
        Remove irrelevant symbols that appear with tesseractOCR
        """
        # df.columns = [col.replace('|', '') for col in df.columns]

        for col in df.columns:
            df[col] = df[col].str.replace("'", "", regex=True)
            df[col] = df[col].str.replace('"', "", regex=True)
            df[col] = df[col].str.replace("\\]", "", regex=True)
            df[col] = df[col].str.replace("\\[", "", regex=True)
            df[col] = df[col].str.replace("\\{", "", regex=True)
            df[col] = df[col].str.replace("\\}", "", regex=True)
        return df

    def convert_df(_self, image_path, df):
        try:
            csv_name = image_path+"_"+timestr+".csv"
            csv_name = re.sub("((.png)|(.jp(e)?g)|(.bmp)|(page_level_images\/))","", csv_name)
            print("Generating CSV....")     
            return df.to_csv(f"{output_dir}/{csv_name}")
        except:
            pass

    def create_dataframe(self, image_path: str, cell_ocr_res: list, max_cols: int, max_rows: int):
        """Create dataframe using list of cell values of the table, also checks for valid header of dataframe
        Args:
            cell_ocr_res: list of strings, each element representing a cell in a table
            max_cols, max_rows: number of columns and rows
        Returns:
            dataframe : final dataframe after all pre-processing
        """

        headers = cell_ocr_res[:max_cols]
        new_headers = uniquify(headers, (f" {x!s}" for x in string.ascii_lowercase))
        counter = 0

        cells_list = cell_ocr_res[max_cols:]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                try:
                    df.iat[nrows, ncols] = str(cells_list[cell_idx])
                    cell_idx += 1
                except IndexError:
                    continue

        ## To check if there are duplicate headers if result of uniquify+col == col
        ## This check removes headers when all headers are empty or if median of header word count is less than 6
        for x, col in zip(string.ascii_lowercase, new_headers):
            if f" {x!s}" == col:
                counter += 1
        header_char_count = [len(col) for col in new_headers]

        df = self.clean_dataframe(df)
        csv = self.convert_df(image_path, df)

        try:
            numkey = df.iloc[0, 0]
        except:
            numkey = str(0)

        return df

    async def start_process(
        self,
        image_path: str,
        TD_THRESHOLD,
        TSR_THRESHOLD,
        OCR_THRESHOLD,
        padd_top,
        padd_left,
        padd_bottom,
        padd_right,
        delta_xmin,
        delta_ymin,
        delta_xmax,
        delta_ymax,
        expand_rowcol_bbox_top,
        expand_rowcol_bbox_bottom,
    ):
        """
        Initiates process of generating pandas dataframes from raw pdf-page images

        """
        image = Image.open(image_path).convert("RGB")
        probas, bboxes_scaled = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)
        print("BBOXES_SCALED: ", bboxes_scaled)
        if bboxes_scaled.nelement() == 0:
            print(f"No table found in the PDF's page image: {image_path}")
            return ""

        self.plot_results_detection(
            table_detection_model,
            image,
            probas,
            bboxes_scaled,
            delta_xmin,
            delta_ymin,
            delta_xmax,
            delta_ymax,
        )
        cropped_img_list = self.crop_tables(
            image_path, image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax
        )

        for idx, unpadded_table in enumerate(cropped_img_list):
            table = self.add_padding(
                unpadded_table, padd_top, padd_right, padd_bottom, padd_left
            )

            probas, bboxes_scaled = table_struct_recog(
                table, THRESHOLD_PROBA=TSR_THRESHOLD
            )
            rows, cols = self.generate_structure(
                table_recognition_model,
                table,
                probas,
                bboxes_scaled,
                expand_rowcol_bbox_top,
                expand_rowcol_bbox_bottom,
            )

            rows, cols = self.sort_table_featuresv2(rows, cols)
            master_row, cols = self.individual_table_featuresv2(table, rows, cols)

            cells_img, max_cols, max_rows = self.object_to_cellsv2(
                master_row,
                cols,
                expand_rowcol_bbox_top,
                expand_rowcol_bbox_bottom,
                padd_left,
            )

            sequential_cell_img_list = []
            for k, img_list in cells_img.items():
                for img in img_list:
                    sequential_cell_img_list.append(
                        pytess(cell_pil_img=img, threshold=OCR_THRESHOLD)
                    )

            cell_ocr_res = await asyncio.gather(*sequential_cell_img_list)

            self.create_dataframe(image_path, cell_ocr_res, max_cols, max_rows)


def convert_pdfs_to_images(input_dir, img_dir):
    print("Converting PDFs to Page level Images..")
    start_time=time.time()

    for file in os.listdir(input_dir):
        images = convert_from_path(os.path.abspath(os.path.join(input_dir,file)))

        for i, image in enumerate(images):
            fname = "image"+str(i)+'.jpg'
            img = str(os.path.abspath(os.path.join(img_dir,file)))[:-4]+"-"+str(i)+'.jpg'
            image.save(img,"JPEG")

    end_time=time.time()

    print('-'*40)
    print("Conversion complete!!")
    print('-'*40)
    print("Time taken: ",round((end_time-start_time),3),"seconds")


if __name__ == "__main__":
    convert_pdfs_to_images(input_dir, pg_img_dir)

    file_count = 0

    t1=time.perf_counter()
    te = TableExtractionPipeline()
    
    for image in os.listdir(pg_img_dir):
        print('-'*50)
        print("Processing image: ",image)
        file_count+=1
        
        asyncio.run(
            te.start_process(
                os.path.join(pg_img_dir, image),
                TD_THRESHOLD=TD_THRESHOLD,
                TSR_THRESHOLD=TSR_THRESHOLD,
                OCR_THRESHOLD=OCR_THRESHOLD,
                padd_top=padd_top,
                padd_left=padd_left,
                padd_bottom=padd_bottom,
                padd_right=padd_right,
                delta_xmin=delta_xmin,  # add offset to the left of the table
                delta_ymin=delta_ymin,  # add offset to the bottom of the table
                delta_xmax=delta_xmax,  # add offset to the right of the table
                delta_ymax=delta_ymax,  # add offset to the top of the table
                expand_rowcol_bbox_top=expand_rowcol_bbox_top,
                expand_rowcol_bbox_bottom=expand_rowcol_bbox_bottom,
            )
        )

    t2=time.perf_counter()
    print('*-'*25)
    print("\nCOMPLETED...")
    print('*-'*25)
    t2=time.perf_counter()
    print("-" * 40)
    image_ctr = "1 image" if file_count == 1 else f"{file_count} images"
    print(f"Processed {image_ctr}")
    print(f"Execution Time: {t2 - t1} seconds.")
    print("-" * 40)