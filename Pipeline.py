import depthai as dai
import cv2

# --- CONFIGURAZIONE MODELLO E CLASSI ---
BLOB_PATH = r"C:\Users\User\Desktop\PROGETTO EDGE MACHINE LEARNING\bestprova.blob"

LABELS = ['Circular Cover', 'Pothole', 'Rectangular Cover', 'Speed Bump']

# Colori: Rosso, Blu, Ciano, Giallo
COLORS = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
# ---------------------------------------

pipeline = dai.Pipeline()

# Configurazione Telecamera
cam = pipeline.create(dai.node.ColorCamera)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) 
cam.setPreviewSize(640, 640)
cam.setInterleaved(False)
cam.setFps(30) 

# Controllo Telecamera
cam_control = pipeline.create(dai.node.XLinkIn)  
cam_control.setStreamName("control")   
cam_control.out.link(cam.inputControl)    

# Rete Neurale YOLO
nn = pipeline.create(dai.node.YoloDetectionNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setConfidenceThreshold(0.4) 
nn.setNumClasses(len(LABELS))  
nn.setCoordinateSize(4)
nn.setIouThreshold(0.5)

# Ancore YOLO
nn.setAnchors([
    10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
    30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
    116.0, 90.0, 156.0, 198.0, 373.0, 326.0
])
nn.setAnchorMasks({
    "side80": [0, 1, 2],
    "side40": [3, 4, 5],
    "side20": [6, 7, 8]
})
nn.setNumInferenceThreads(3)

# Linking
cam.preview.link(nn.input)

xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")
cam.preview.link(xout_cam.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

print("Avvio della connessione con OAK-1-PoE in corso...")

# Avvio!
with dai.Device(pipeline) as device:
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_nn  = device.getOutputQueue(name="nn",  maxSize=4, blocking=False)
    
    print("Pipeline avviata con successo! Premi 'ESC' per chiudere la finestra video.")

    while True:
        in_cam = q_cam.get()
        in_nn = q_nn.get()

        frame = in_cam.getCvFrame()
        detections = in_nn.detections

        for det in detections:
            x1, y1 = int(det.xmin * frame.shape[1]), int(det.ymin * frame.shape[0])
            x2, y2 = int(det.xmax * frame.shape[1]), int(det.ymax * frame.shape[0])
            
            cls = int(det.label)
            conf = det.confidence
            color = COLORS[cls] if cls < len(COLORS) else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{LABELS[cls]} {conf:.2f}"
            
            # Sfondo per il testo
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Road Fault Detection - OAK-1-PoE", frame)
        if cv2.waitKey(1) == 27:
            break