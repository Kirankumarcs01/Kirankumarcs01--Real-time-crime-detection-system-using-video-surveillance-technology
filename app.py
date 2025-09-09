from flask import Flask, redirect, render_template, url_for, request
import sqlite3

import pygame
from A_Recognition import analyse
import os
from flask import Flask, request, render_template
import argparse
import csv
# Flag to track if audio has been played
from playsound import playsound
import platform
import sys
from pathlib import Path
import telepot
bot=telepot.Bot('7138201104:AAG7gvB-AE1Gaw7XBAdbBK-GUR0B8cH6LGE')
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
# UPLOAD_FOLDER = 'uploads'  # Folder to save uploaded files
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('userlog.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

 

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
# Create the uploads directory if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/Live1', methods=['POST'])
def Live1():
    audio_played = True 
    # Check if the file is in the request
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return "No selected file", 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Print the full file path
    print(f"File uploaded to: {file_path}")

    # Call your main function with the file content
    

    # Set the source path manually based on the uploaded file's path
    sources = file_path 
    

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


    from ultralytics.utils.plotting import Annotator, colors, save_one_box

    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (
        LOGGER,
        Profile,
        check_file,
        check_img_size,
        check_imshow,
        check_requirements,
        colorstr,
        cv2,
        increment_path,
        non_max_suppression,
        print_args,
        scale_boxes,
        strip_optimizer,
        xyxy2xywh,
    )
    from utils.torch_utils import select_device, smart_inference_mode
   

  

    @smart_inference_mode()
    def run(
        weights=ROOT / "yolov5s.pt",  # model path or triton URL
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
    ):
        # source = str(source)
        # save_img = not nosave and not source.endswith(".txt")  # save inference images
        # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        # is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        # webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        source = sources # Path to your video file
        
        save_img = not nosave and not source.endswith(".txt")  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        webcam = False  # Explicitly disable webcam since we are using a file
        screenshot = source.lower().startswith("screen")
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / "predictions.csv"

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not csv_path.is_file():
                        writer.writeheader()
                    writer.writerow(data)


                    
            pygame.mixer.init()

# Define the path to your MP3 file
            sound_path = 'fx-a-terrifying-and-alarming-siren-screaming-in-the-night-254556 (2).mp3'


            # Function to play the sound
            def play_sound():
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()

            # Flag to check if sound is playing
            is_playing = False


            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        # If the sound is not playing, play it
                    if not pygame.mixer.music.get_busy():  # Check if the sound is not currently playing
                        play_sound()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            bot.sendMessage('1847624604',str("weapon detected"))
                            cv2.imwrite("frame1.jpg",im0)
                            bot.sendPhoto("1847624604",photo=open("frame1.jpg",'rb'))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
                            
                          
                # Stream results
                im0 = annotator.result()
                cv2.imshow('Detection Result', im0)
           
                key = cv2.waitKey(1)  # 1 millisecond delay to capture key press
                if key == ord('q'):  # Check if 'q' is pressed
                    print("Stopping detection and sound...")
                    pygame.mixer.music.stop()  # Stop the sound
                    cv2.destroyAllWindows()  # Close the OpenCV window

                    # Redirect to userlog.html after pressing 'q'
                    return redirect(url_for('userlog'))
                if view_img:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    

    def parse_opt(File):
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
        parser.add_argument("--source", type=str, default=File, help="file/dir/URL/glob/screen/0(webcam)")
        parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
        parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
        parser.add_argument("--conf-thres", type=float, default=0.45, help="confidence threshold")
        parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
        parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
        parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
        parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
        parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
        parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
        parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
        parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
        parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
        parser.add_argument("--augment", action="store_true", help="augmented inference")
        parser.add_argument("--visualize", action="store_true", help="visualize features")
        parser.add_argument("--update", action="store_true", help="update all models")
        parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
        parser.add_argument("--name", default="exp", help="save results to project/name")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
        parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
        parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
        parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
        parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
        parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt


    def main(opt):
        check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
        run(**vars(opt))


    f = open('temp.txt', 'r')
    File = f.read()
    f.close()
    opt = parse_opt(File)
    main(opt)
    return render_template('userlog.html') 



@app.route('/Live2', methods=['POST'])
def Live2():
    # Check if the file is in the request
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return "No selected file", 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Print the full file path
    print(f"File uploaded to: {file_path}")

    # Call your main function with the file content
    

    # Set the source path manually based on the uploaded file's path
    sources = file_path 
    

        
  

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


    from ultralytics.utils.plotting import Annotator, colors, save_one_box

    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (
        LOGGER,
        Profile,
        check_file,
        check_img_size,
        check_imshow,
        check_requirements,
        colorstr,
        cv2,
        increment_path,
        non_max_suppression,
        print_args,
        scale_boxes,
        strip_optimizer,
        xyxy2xywh,
    )
    from utils.torch_utils import select_device, smart_inference_mode


    @smart_inference_mode()
    def run(
        weights=ROOT / "yolov5s.pt",  # model path or triton URL
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
    ):
        # source = str(source)
        # save_img = not nosave and not source.endswith(".txt")  # save inference images
        # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        # is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        # webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        
        source = sources # Path to your video file
        save_img = not nosave and not source.endswith(".txt")  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        webcam = False  # Explicitly disable webcam since we are using a file
        screenshot = source.lower().startswith("screen")
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / "predictions.csv"

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not csv_path.is_file():
                        writer.writeheader()
                    writer.writerow(data)



            pygame.mixer.init()

# Define the path to your MP3 file
            sound_path = 'fx-a-terrifying-and-alarming-siren-screaming-in-the-night-254556 (2).mp3'


            # Function to play the sound
            def play_sound():
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()

            # Flag to check if sound is playing
            is_playing = False

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    if not pygame.mixer.music.get_busy():  # Check if the sound is not currently playing
                        play_sound()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            bot.sendMessage('1847624604',str("accident detected"))
                            cv2.imwrite("frame1.jpg",im0)
                            bot.sendPhoto("1847624604",photo=open("frame1.jpg",'rb'))
                        
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                # Stream results
                im0 = annotator.result()
                cv2.imshow('Detection Result', im0)
           
                key = cv2.waitKey(1)  # 1 millisecond delay to capture key press
                
                if key == ord('q'):  # Check if 'q' is pressed
                    print("Stopping detection and sound...")
                    pygame.mixer.music.stop()  # Stop the sound
                    cv2.destroyAllWindows()  # Close the OpenCV window

                    # Redirect to userlog.html after pressing 'q'
                    return redirect(url_for('userlog'))
                if view_img:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


    def parse_opt(File):
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "ACCIDENT.pt", help="model path or triton URL")
        parser.add_argument("--source", type=str, default=File, help="file/dir/URL/glob/screen/0(webcam)")
        parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
        parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
        parser.add_argument("--conf-thres", type=float, default=0.45, help="confidence threshold")
        parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
        parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
        parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
        parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
        parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
        parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
        parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
        parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
        parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
        parser.add_argument("--augment", action="store_true", help="augmented inference")
        parser.add_argument("--visualize", action="store_true", help="visualize features")
        parser.add_argument("--update", action="store_true", help="update all models")
        parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
        parser.add_argument("--name", default="exp", help="save results to project/name")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
        parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
        parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
        parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
        parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
        parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt


    def main(opt):
        check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
        run(**vars(opt))


    f = open('temp.txt', 'r')
    File = f.read()
    f.close()
    opt = parse_opt(File)
    main(opt)
    return render_template('userlog.html') 






@app.route('/Live3', methods=['POST'])
def Live3():
    # Check if the file is in the request
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return "No selected file", 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Print the full file path
    print(f"File uploaded to: {file_path}")

    # Call your main function with the file content
    

    # Set the source path manually based on the uploaded file's path
    sources = file_path 
    

        
    

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


    from ultralytics.utils.plotting import Annotator, colors, save_one_box

    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (
        LOGGER,
        Profile,
        check_file,
        check_img_size,
        check_imshow,
        check_requirements,
        colorstr,
        cv2,
        increment_path,
        non_max_suppression,
        print_args,
        scale_boxes,
        strip_optimizer,
        xyxy2xywh,
    )
    from utils.torch_utils import select_device, smart_inference_mode


    @smart_inference_mode()
    def run(
        weights=ROOT / "yolov5s.pt",  # model path or triton URL
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
    ):
        source = sources # Path to your video file
        save_img = not nosave and not source.endswith(".txt")  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        webcam = False  # Explicitly disable webcam since we are using a file
        screenshot = source.lower().startswith("screen")
        
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names= {0: 'explotion'}
        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / "predictions.csv"

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not csv_path.is_file():
                        writer.writeheader()
                    writer.writerow(data)

            pygame.mixer.init()

# Define the path to your MP3 file
            sound_path = 'fx-a-terrifying-and-alarming-siren-screaming-in-the-night-254556 (2).mp3'


            # Function to play the sound
            def play_sound():
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()

            # Flag to check if sound is playing
            is_playing = False

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    if not pygame.mixer.music.get_busy():  # Check if the sound is not currently playing
                        play_sound()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            bot.sendMessage('5850081723',str("explosion happen red alert"))
                            cv2.imwrite("frame1.jpg",im0)
                            bot.sendPhoto("5850081723",photo=open("frame1.jpg",'rb'))
                    
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                # Stream results
                im0 = annotator.result()
                cv2.imshow('Detection Result', im0)
           
                key = cv2.waitKey(1)  # 1 millisecond delay to capture key press
                
                if key == ord('q'):  # Check if 'q' is pressed
                    print("Stopping detection and sound...")
                    pygame.mixer.music.stop()  # Stop the sound
                    cv2.destroyAllWindows()  # Close the OpenCV window

                    # Redirect to userlog.html after pressing 'q'
                    return redirect(url_for('userlog'))
                if view_img:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


    def parse_opt(File):
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "fire.pt", help="model path or triton URL")
        parser.add_argument("--source", type=str, default=File, help="file/dir/URL/glob/screen/0(webcam)")
        parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
        parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
        parser.add_argument("--conf-thres", type=float, default=0.45, help="confidence threshold")
        parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
        parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
        parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
        parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
        parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
        parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
        parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
        parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
        parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
        parser.add_argument("--augment", action="store_true", help="augmented inference")
        parser.add_argument("--visualize", action="store_true", help="visualize features")
        parser.add_argument("--update", action="store_true", help="update all models")
        parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
        parser.add_argument("--name", default="exp", help="save results to project/name")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
        parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
        parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
        parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
        parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
        parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt


    def main(opt):
        check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
        run(**vars(opt))


    f = open('temp.txt', 'r')
    File = f.read()
    f.close()
    opt = parse_opt(File)
    main(opt)
    return render_template('userlog.html') 



if __name__ == '__main__':
    audio_played = False
    app.run(debug=True)