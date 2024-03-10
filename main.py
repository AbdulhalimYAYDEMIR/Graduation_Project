#References:                                                                              #
#https://github.com/ibaiGorordo/AWR1642-Read-Data-Python-MMWAVE-SDK-2/tree/heatMapDevelop #
#https://www.youtube.com/watch?v=jWxNfb7Hng8&t=612s&pp=ygURcHlxdDUgYmVzdCBkZXNpZ24%3D     #
###########################################################################################

from PyQt5.QtWidgets import QApplication, QMainWindow ,QListWidgetItem
from ui.main_window_ui import Ui_MainWindow
from pages_functions.graphic import Graphic
from pages_functions.cfg import Cfg



import os
import shutil
import numpy as np
import matplotlib.pyplot as plt



class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui_cfg = Cfg()
        self.ui.setupUi(self)


        self.CFG_btn = self.ui.pushButton
        self.HeatMapv1_btn =self.ui.pushButton_2
        self.HeatMapv2_btn = self.ui.pushButton_3
        self.HeatMapv3_btn = self.ui.pushButton_4
        self.Range_Profile_btn = self.ui.pushButton_5
        self.HeatMap_Max_Points_btn = self.ui.pushButton_6
        self.Search_btn = self.ui.pushButton_9



        self.Search_btn.clicked.connect(lambda: self.main_function(str(self.ui.lineEdit.text()),str(self.ui.lineEdit_4.text()),str(self.ui.lineEdit_5.text())))



        self.menu_btns_dict = {
            self.CFG_btn: self.ui_cfg,
            self.HeatMapv1_btn: Graphic,
            self.HeatMapv2_btn: Graphic,
            self.HeatMapv3_btn: Graphic,
            self.Range_Profile_btn: Graphic,
            self.HeatMap_Max_Points_btn: Graphic,

        }

        self.show_home_window()
        self.CFG_btn.clicked.connect(self.show_selected_window)
        self.HeatMapv1_btn.clicked.connect(self.show_selected_window)
        self.HeatMapv2_btn.clicked.connect(self.show_selected_window)
        self.HeatMapv3_btn.clicked.connect(self.show_selected_window)
        self.Range_Profile_btn.clicked.connect(self.show_selected_window)
        self.HeatMap_Max_Points_btn.clicked.connect(self.show_selected_window)


    def show_home_window(self):
        result = self.open_tab_flag(self.CFG_btn.text())
        self.set_btn_checked(self.CFG_btn)

        if result[0]:
            self.ui.tabWidget.setCurrentIndex(result[1])
        else:
            tab_title = self.CFG_btn.text()
            curIndex = self.ui.tabWidget.addTab(self.ui_cfg,tab_title)
            self.ui.tabWidget.setCurrentIndex(curIndex)
            self.ui.tabWidget.setVisible(True)


    def set_btn_checked(self,btn):
        for button in self.menu_btns_dict.keys():
            if button != btn:
                button.setChecked(False)
            else:
                button.setChecked(True)


    def show_selected_window(self):
        button = self.sender()
        result = self.open_tab_flag(button.text())
        self.set_btn_checked(button)

        if result[0]:
            self.ui.tabWidget.setCurrentIndex(result[1])
        else:
            tab_title = button.text()
            curIndex = self.ui.tabWidget.addTab(self.menu_btns_dict[button](),tab_title)
            self.ui.tabWidget.setCurrentIndex(curIndex)
            self.ui.tabWidget.setVisible(True)



    def open_tab_flag(self,btn_text):
        open_tab_count = self.ui.tabWidget.count()

        for i in range(open_tab_count):
            tab_title = self.ui.tabWidget.tabText(i)
            if tab_title == btn_text:
                return True, i
            else:
                continue

        return False,







    # -------------------------------------DEFİNES---------------------------

    # Number of antennas used
    global numRxAnt
    numRxAnt = 4

    global numTxAnt
    numTxAnt = 2

    global numTxAzimAnt
    numTxAzimAnt = 2

    global NUM_ANGLE_BINS
    NUM_ANGLE_BINS = 64

    # Folder containing dat files
    global directory
    directory = (os.getcwd() + "/dat_files").replace("\\", "/")

    global configFileName
    configFileName = (os.getcwd() + "/xwr16xx_BestRangeResolution_UpdateRate_1.cfg").replace("\\", "/")



    # ----------------------------FUNCTIONS--------------------------------------

    # Magnitude to Desibel Converter
    def mag2db(self,magnitude):
        for i in range(256):
            for j in range(63):
                if magnitude[i, j] == 0:
                    magnitude[i, j] = 1

        return 20 * np.log10(magnitude)

    # ------------------------------------------------

    # Desibel to Magnitude Converter
    def db2mag(self,decibel):
        return 10 ** (decibel / 20)

    # ------------------------------------------------

    # Function to parse the data inside the configuration file
    def parseConfigFile(self,configFileName):
        configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:

            # Split the line
            splitWords = i.split(" ")

            # Hard code the number of antennas, change if other configuration is used
            Virtual_Antenna_Azim = numRxAnt * numTxAnt

            # Get the information about the profile configuration
            if "profileCfg" in splitWords[0]:
                startFreq = int(float(splitWords[2]))
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = float(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1;

                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;

                digOutSampleRate = int(splitWords[11]);

            # Get the information about the frame configuration
            elif "frameCfg" in splitWords[0]:

                chirpStartIdx = int(splitWords[1]);
                chirpEndIdx = int(splitWords[2]);
                numLoops = int(splitWords[3]);
                numFrames = int(splitWords[4]);
                framePeriodicity = int(splitWords[5]);

        # Combine the read data to obtain the configuration parameters
        numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
        configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
        configParameters["numRangeBins"] = numAdcSamplesRoundTo2
        configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
        configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
        configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
        configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
        configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
        configParameters["Virtual_Antenna_Azim"] = Virtual_Antenna_Azim


        self.ui_cfg.ui.listWidget.addItem("---------------------CFG-------------------")
        self.ui_cfg.ui.listWidget.addItem(("numDopplerBins : " + str(configParameters["numDopplerBins"])))
        self.ui_cfg.ui.listWidget.addItem(("numRangeBins : " + str(configParameters["numRangeBins"])))
        self.ui_cfg.ui.listWidget.addItem(("rangeResolutionMeters : " + str(configParameters["rangeResolutionMeters"])))
        self.ui_cfg.ui.listWidget.addItem(("rangeIdxToMeters : " + str(configParameters["rangeIdxToMeters"])))
        self.ui_cfg.ui.listWidget.addItem(("dopplerResolutionMps : " + str(configParameters["dopplerResolutionMps"])))
        self.ui_cfg.ui.listWidget.addItem(("maxRange : " + str(configParameters["maxRange"])))
        self.ui_cfg.ui.listWidget.addItem(("maxVelocity : " + str(configParameters["maxVelocity"])))
        self.ui_cfg.ui.listWidget.addItem(("Virtual_Antenna_Azim : " + str(configParameters["Virtual_Antenna_Azim"])))
        self.ui_cfg.ui.listWidget.addItem(("numAdcSamples : " + str(numAdcSamples)))
        self.ui_cfg.ui.listWidget.addItem(("numAdcSamplesRoundTo2 : " + str(numAdcSamplesRoundTo2)))
        self.ui_cfg.ui.listWidget.addItem(("Virtual_Antenna_Azim : " + str(configParameters["Virtual_Antenna_Azim"])))
        self.ui_cfg.ui.listWidget.addItem(("numRxAnt : " + str(numRxAnt)))
        self.ui_cfg.ui.listWidget.addItem(("numTxAnt : " + str(numTxAnt)))
        self.ui_cfg.ui.listWidget.addItem(("numTxAzimAnt : " + str(numTxAzimAnt)))
        self.ui_cfg.ui.listWidget.addItem(("NUM_ANGLE_BINS : " + str(NUM_ANGLE_BINS)))
        self.ui_cfg.ui.listWidget.addItem(("configFileName : " + str(configFileName)))


        return configParameters

    # ------------------------------------------------

    # Funtion to read and parse the incoming data
    def readAndParseData16xx(self,data_2, size_2, configParameters):
        byteBuffer = np.zeros(2 ** 25, dtype='uint8')
        byteBufferLength = 0;

        # Constants
        OBJ_STRUCT_SIZE_BYTES = 12
        BYTE_VEC_ACC_MAX_SIZE = 2 ** 25
        MMWDEMO_UART_MSG_DETECTED_POINTS = 1
        MMWDEMO_UART_MSG_RANGE_PROFILE = 2
        MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
        MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
        maxBufferSize = 2 ** 25
        magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

        # Initialize variables
        magicOK = 0  # Checks if magic number has been read
        dataOK = 0  # Checks if the data has been read correctly
        frameNumber = 0
        detObj = {}
        azimMapObject = {}
        result = 1

        # Read data as byte and Length of it
        data_2 = np.frombuffer(data_2, dtype='uint8')
        byteVec = data_2
        byteCount = size_2

        # Check that the buffer is not full, and then add the data to the buffer
        if (byteBufferLength + byteCount) < maxBufferSize:
            byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
            byteBufferLength = byteBufferLength + byteCount

        # Check that the buffer has some data, for example 16
        if byteBufferLength > 16:

            # Check for all possible locations of the magic word
            possibleLocs = np.where(byteBuffer == magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = byteBuffer[loc:loc + 8]
                if np.all(check == magicWord):
                    startIdx.append(loc)

            # Check that startIdx is not empty
            if startIdx:

                # Remove the data before the first start index
                if startIdx[0] > 0:
                    byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                    byteBufferLength = byteBufferLength - startIdx[0]

                # Check that there have no errors with the byte buffer length
                if byteBufferLength < 0:
                    byteBufferLength = 0

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Read the total packet length
                totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)  # True

                # Check that all the packet has been read
                if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                    magicOK = 1

                # if incoming data is less than the packet size, exit this function
                if startIdx[0] + totalPacketLen > byteCount:
                    self.ui_cfg.ui.listWidget.addItem("!!! This Frame has not enough length for reading !!!")
                    result = 0
                    return result, dataOK, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        # If magicOK is equal to 1 then process the message
        if magicOK:
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Initialize the pointer index
            idX = 0

            # Read the header
            magicNumber = byteBuffer[idX:idX + 8]
            idX += 8
            version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
            self.ui_cfg.ui.listWidget.addItem(("TotalPacketLen : " + str(totalPacketLen)))
            idX += 4
            platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
            self.ui_cfg.ui.listWidget.addItem(("NumDetectedObj : " + str(numDetectedObj)))
            idX += 4
            numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
            self.ui_cfg.ui.listWidget.addItem(("NumTLVs : " + str(numTLVs)))
            idX += 4
            subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4

            idX2 = idX

            # Read the TLV messages
            for tlvIdx in range(numTLVs):

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Check the header of the TLV message
                tlv_type = np.matmul(byteBuffer[idX2:idX2 + 4], word)
                self.ui_cfg.ui.listWidget.addItem(("TlvType : " + str(tlv_type)))
                idX2 += 4
                tlv_length = np.matmul(byteBuffer[idX2:idX2 + 4], word)
                idX2 += 4
                self.ui_cfg.ui.listWidget.addItem(("TlvLen : " + str(tlv_length)))

                idX = idX2
                idX2 += tlv_length

                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                    # word array to convert 4 bytes to a 16 bit number
                    word = [1, 2 ** 8]
                    tlv_numObj = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    tlv_xyzQFormat = 2 ** np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2

                    # Initialize the arrays
                    rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                    dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                    peakVal = np.zeros(tlv_numObj, dtype='int16')
                    x = np.zeros(tlv_numObj, dtype='int16')
                    y = np.zeros(tlv_numObj, dtype='int16')
                    z = np.zeros(tlv_numObj, dtype='int16')

                    for objectNum in range(numDetectedObj):  # burada tlv_numObj vardı 49434 değeri alıyodu numDetectedObj bunu yazım 7 alıyor
                        # Read the data for each object
                        rangeIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        peakVal[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        x[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        y[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        z[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2

                    # Make the necessary corrections and calculate the rest of the data
                    rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                    dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)] = dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)] - 65535
                    dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                    # x[x > 32767] = x[x > 32767] - 65536
                    # y[y > 32767] = y[y > 32767] - 65536
                    # z[z > 32767] = z[z > 32767] - 65536
                    # x = x / tlv_xyzQFormat
                    # y = y / tlv_xyzQFormat
                    # z = z / tlv_xyzQFormat

                    # Store the data in the detObj dictionary
                    detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                              "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}





                elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                    word = [1, 2 ** 8]
                    Range_Profile = np.zeros(configParameters["numRangeBins"], dtype='int16')
                    rangeAxis = np.arange(0, configParameters['numRangeBins'] * configParameters["rangeIdxToMeters"],configParameters["rangeIdxToMeters"])

                    for i in range(configParameters["numRangeBins"]):
                        Range_Profile[i] = np.matmul(byteBuffer[idX:idX + 2], word)
                        Range_Profile[i] = (Range_Profile[i] * ((2 ** np.ceil(np.log2(configParameters["Virtual_Antenna_Azim"]))) / (512.0 * configParameters["Virtual_Antenna_Azim"])) * (20.0 * np.log10(2))) + (20 * np.log10(32.0 / configParameters["numRangeBins"]))
                        idX += 2



                elif tlv_type == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP:

                    numBytes = numTxAzimAnt * numRxAnt * configParameters["numRangeBins"] * 4;

                    q = byteBuffer[idX:idX + numBytes]

                    idX += numBytes
                    qrows = numTxAzimAnt * numRxAnt
                    qcols = configParameters["numRangeBins"]

                    # first data imaginary part, second data real part reference: UART DATA OUTPUT FORMAT
                    imaginary = q[::4] + q[1::4] * 256
                    real = q[2::4] + q[3::4] * 256

                    real = real.astype(np.int16)
                    imaginary = imaginary.astype(np.int16)

                    q = real + 1j * imaginary

                    q = np.reshape(q, (qrows, qcols), order="F")

                    Q = np.fft.fft(q, NUM_ANGLE_BINS, axis=0)
                    QQ = np.fft.fftshift(abs(Q), axes=0);
                    QQ = QQ.T

                    QQ = QQ[:, 1:]
                    QQ = np.fliplr(QQ)

                    theta = np.rad2deg(np.arcsin(np.array(range(-NUM_ANGLE_BINS // 2 + 1, NUM_ANGLE_BINS // 2)) * (2 / NUM_ANGLE_BINS)))
                    rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]

                    posX = np.outer(rangeArray.T, np.sin(np.deg2rad(theta)))
                    posY = np.outer(rangeArray.T, np.cos(np.deg2rad(theta)))

                    # Store the data in the azimMapObject dictionary
                    azimMapObject = {"posX": posX, "posY": posY, "range": rangeArray, "theta": theta, "heatMap": QQ}

                    dataOK = 1




                elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:
                    numBytes = configParameters["numDopplerBins"] * configParameters["numRangeBins"] * 2
                    rangeDoppler = byteBuffer[idX:idX + numBytes]

                    rangeDoppler = rangeDoppler[::2] + rangeDoppler[1::2] * 256

                    rangeDoppler = np.reshape(rangeDoppler,(configParameters["numDopplerBins"], configParameters["numRangeBins"]))
                    rangeDoppler = np.concatenate((byteBuffer[(byteBufferLength + 1) // 2:], byteBuffer[:(byteBufferLength + 1) // 2]))

                    idX += tlv_length
                    # NOT FINISHED, CONTINUE IF NECESSARY

            # Remove already processed data, This process is not used for this algorithm. byteBuffer and byteBufferLength are updated at for loop in main part
            if idX > 0 and dataOK == 1:
                shiftSize = idX

                byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
                byteBufferLength = byteBufferLength - shiftSize

                # Check that there are no errors with the buffer length
                if byteBufferLength < 0:
                    byteBufferLength = 0

        return result, dataOK, frameNumber, detObj, azimMapObject, startIdx[0], totalPacketLen, Range_Profile, rangeAxis, QQ, theta, rangeArray

    # ------------------------------------------------

    # Funtion to update the data and display in the plot
    def update(self,data_1, size_1,configParameters):

        dataOk = 0
        global detObj
        global azimMapObject
        x = []
        y = []

        # Read and parse the received data
        result_, dataOk, frameNumber, detObj, azimMapObject, startIdx_, totalPacketLen_, Range_Profile_, rangeAxis_, QQ_, theta_, rangeArray_ = self.readAndParseData16xx(data_1, size_1, configParameters)

        return result_, dataOk, startIdx_, totalPacketLen_, Range_Profile_, rangeAxis_, QQ_, theta_, rangeArray_, azimMapObject

    # ------------------------------------------------

    # Sort downloaded dat files by creation date
    def file_info(self,directory):
        file_list = []
        file_list_new = []

        for i in os.listdir(directory):
            a = os.stat(os.path.join(directory, i)).st_ctime
            file_list.append(a)

        file_list_sorted = sorted(file_list)

        for i in range(len(file_list_sorted)):
            file_list_new.append(os.listdir(directory)[file_list.index(file_list_sorted[i])])

        return file_list_new

    def main_function(self,text,colorbar_min,colorbar_max):

        self.ui_cfg.ui.listWidget.clear()

        self.ui_cfg.ui.listWidget.addItem(("colorbar_min : " + str(colorbar_min)))
        self.ui_cfg.ui.listWidget.addItem(("colorbar_max : " + str(colorbar_max)))
        self.ui_cfg.ui.listWidget.addItem(" ")


        if (colorbar_min != "") & (colorbar_max != ""):
            colorbar_min = int(colorbar_min)
            colorbar_max = int(colorbar_max)

        elif (colorbar_min == "") & (colorbar_max != ""):
            colorbar_min = 0
            colorbar_max = int(colorbar_max)

        elif (colorbar_min != "") & (colorbar_max == ""):
            colorbar_min = int(colorbar_min)
            colorbar_max = 0
        else:
            colorbar_min = 0
            colorbar_max = 0



        # More than one input can be entered at once
        dat_files_number = np.fromstring(text, dtype=int,sep=' ')  # list(map(int, input('Enter .dat files number : ').split()))

        file_list = self.file_info(directory)

        # Foldering procedures for outputs
        if not (os.path.exists((os.getcwd() + "/figures").replace("\\", "/"))):
            os.mkdir((os.getcwd() + "/figures").replace("\\", "/"))
        else:
            shutil.rmtree((os.getcwd() + "/figures").replace("\\", "/"))
            os.mkdir((os.getcwd() + "/figures").replace("\\", "/"))

        os.mkdir((os.getcwd() + "/figures" + "/heatmap_x_y").replace("\\", "/"))
        os.mkdir((os.getcwd() + "/figures" + "/heatmap_range_theta_v1").replace("\\", "/"))
        os.mkdir((os.getcwd() + "/figures" + "/range_profile").replace("\\", "/"))
        os.mkdir((os.getcwd() + "/figures" + "/heatmap_range_theta_v2").replace("\\", "/"))

        # Keeps record of max db of each measurement
        max_db = []
        max_db_axis = []

        # Each dat files is parsed
        for i in dat_files_number:

            # find and read the relevant dat file
            fp = open(os.path.join(directory, file_list[(len(file_list) - 1) - (len(file_list) - i)]).replace("\\", "/"),'rb')
            readNumBytes = os.path.getsize(os.path.join(directory, file_list[(len(file_list) - 1) - (len(file_list) - i)]).replace("\\", "/"))
            allBinData = fp.read()
            fp.close()

            # initializes
            totalBytesParsed = 0
            numFramesParsed = 0
            Range_Profile_Total = 0
            Range_Profile_Average = 0
            QQ_Total = 0
            QQ_Average = 0

            # Get the configuration parameters from the configuration file
            self.ui_cfg.ui.listWidget.addItem(" ")
            configParameters = self.parseConfigFile(configFileName)

            self.ui_cfg.ui.listWidget.addItem(" ")
            self.ui_cfg.ui.listWidget.addItem("---------------------FILE---------------------")
            self.ui_cfg.ui.listWidget.addItem(os.path.join(directory, file_list[(len(file_list) - 1) - (len(file_list) - i)]).replace("\\", "/"))
            self.ui_cfg.ui.listWidget.addItem(("ReadNumBytes : " + str(readNumBytes)))


            # Parsing loop
            detObj = {}
            azimMapObject = {}
            frameData = {}
            currentIndex = 0
            while (totalBytesParsed < readNumBytes):

                try:
                    self.ui_cfg.ui.listWidget.addItem("----------------------------------------------")
                    # Update the data and check if the data is okay
                    result, dataOk, headerStartIndex, totalPacketNumBytes, Range_Profile, rangeAxis, QQ, theta, rangeArray, azimMapObject = self.update(allBinData[totalBytesParsed::1], readNumBytes - totalBytesParsed,configParameters)
                    totalBytesParsed += (headerStartIndex + totalPacketNumBytes)

                    if result == 0:
                        break

                    if dataOk and result:
                        # Store the current frame into frameData
                        frameData[currentIndex] = detObj
                        currentIndex += 1
                        self.ui_cfg.ui.listWidget.addItem(("currentIndex : " + str(currentIndex)))
                        self.ui_cfg.ui.listWidget.addItem(("---------------------Frame" + str(currentIndex) + "-----------------"))

                        rangeAxiss = rangeAxis
                        thetaa = theta
                        rangeArrayy = rangeArray
                        azimMapObjectt = azimMapObject

                        QQ_Total += QQ
                        Range_Profile_Total += Range_Profile





                # Stop the program and close everything if Ctrl + c is pressed
                except KeyboardInterrupt:
                    break

            # average of all frames
            Range_Profile_Average = Range_Profile_Total / currentIndex
            QQ_Average = QQ_Total / currentIndex



            # ------------- Figure Creations and Saving------------------------

            # HEAT MAP on X-Y AXİS (Heatmap version-1)
            fig1, ax1 = plt.subplots()

            if not(self.ui.pushButton_10.isChecked()):

                if (colorbar_min == 0) & (colorbar_max == 0):
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], self.mag2db(QQ_Average),100, cmap="jet")  # vmin=-30,vmax=90
                elif (colorbar_min == 0) & (colorbar_max != 0):
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], self.mag2db(QQ_Average),100, cmap="jet", vmax=colorbar_max)  # vmin=-30,vmax=90
                elif (colorbar_min != 0) & (colorbar_max == 0):
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], self.mag2db(QQ_Average),100, cmap="jet", vmin=colorbar_min)  # vmin=-30,vmax=90
                else:
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], self.mag2db(QQ_Average),100, cmap="jet", vmin=colorbar_min, vmax=colorbar_max)  # vmin=-30,vmax=90

                plt.text(6.8, 6.7,"Max dB : " + str(format(np.max(self.mag2db(QQ_Average)), '.3f')) + "\n" + "Min  dB : " + str(format(np.min(self.mag2db(QQ_Average)), '.3f')))
                cbar1 = fig1.colorbar(Heat_Map1)
                cbar1.set_label('Desibel (dB)')
            else:
                if (colorbar_min == 0) & (colorbar_max == 0):
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], QQ_Average,100, cmap="jet")  # vmin=-30,vmax=90
                elif (colorbar_min == 0) & (colorbar_max != 0):
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], QQ_Average,100, cmap="jet", vmax=colorbar_max)  # vmin=-30,vmax=90
                elif (colorbar_min != 0) & (colorbar_max == 0):
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], QQ_Average,100, cmap="jet", vmin=colorbar_min)  # vmin=-30,vmax=90
                else:
                    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], QQ_Average,100, cmap="jet", vmin=colorbar_min, vmax=colorbar_max)  # vmin=-30,vmax=90

                plt.text(6.8, 6.7,"Max Mag: " + str(format(np.max((QQ_Average)), '.3f')) + "\n" + "Min Mag : " + str(format(np.min((QQ_Average)), '.3f')))
                cbar1 = fig1.colorbar(Heat_Map1)
                cbar1.set_label('Magnitude (Mag)')

            plt.xlabel("Horizontal, x axis (meters)")
            plt.ylabel("Longitudinal, y axis (meters)")
            title = "Range/Azimuth Heat Map                                "
            id = str(i)
            plt.title(title + id)
            plt.savefig((os.getcwd() + "/figures" + "/heatmap_x_y/" + str(i) + ".png").replace("\\", "/"))
            plt.close("all")







            # HEAT MAP on RANGE-THETA AXİS (Heatmap version-2)
            fig2, ax2 = plt.subplots()
            X, Y = np.meshgrid(azimMapObjectt["theta"], azimMapObjectt["range"])

            if not(self.ui.pushButton_10.isChecked()):

                if (colorbar_min == 0) & (colorbar_max == 0):
                    Heat_Map2 = plt.contourf(X, Y, self.mag2db(QQ_Average), 100, cmap="jet")
                elif (colorbar_min == 0) & (colorbar_max != 0):
                    Heat_Map2 = plt.contourf(X, Y, self.mag2db(QQ_Average), 100, cmap="jet", vmax=colorbar_max)
                elif (colorbar_min != 0) & (colorbar_max == 0):
                    Heat_Map2 = plt.contourf(X, Y, self.mag2db(QQ_Average), 100, cmap="jet", vmin=colorbar_min)
                else:
                    Heat_Map2 = plt.contourf(X, Y, self.mag2db(QQ_Average), 100, cmap="jet", vmin=colorbar_min,vmax=colorbar_max)

                plt.text(84, 6.7,"Max dB : " + str(format(np.max(self.mag2db(QQ_Average)), '.3f')) + "\n" + "Min  dB : " + str(format(np.min(self.mag2db(QQ_Average)), '.3f')))
                cbar2 = fig2.colorbar(Heat_Map2)
                cbar2.set_label('Desibel (dB)')
            else:
                if (colorbar_min == 0) & (colorbar_max == 0):
                    Heat_Map2 = plt.contourf(X, Y, QQ_Average, 100, cmap="jet")
                elif (colorbar_min == 0) & (colorbar_max != 0):
                    Heat_Map2 = plt.contourf(X, Y, QQ_Average, 100, cmap="jet", vmax=colorbar_max)
                elif (colorbar_min != 0) & (colorbar_max == 0):
                    Heat_Map2 = plt.contourf(X, Y, QQ_Average, 100, cmap="jet", vmin=colorbar_min)
                else:
                    Heat_Map2 = plt.contourf(X, Y, QQ_Average, 100, cmap="jet", vmin=colorbar_min,vmax=colorbar_max)

                plt.text(84, 6.7,"Max Mag: " + str(format(np.max((QQ_Average)), '.3f')) + "\n" + "Min Mag : " + str(format(np.min((QQ_Average)), '.3f')))
                cbar2 = fig2.colorbar(Heat_Map2)
                cbar2.set_label('Magnitude (Mag)')

            plt.xlabel("Angle axis (degrees)")
            plt.ylabel("Range axis (meters)")
            title = "Range/Azimuth Heat Map                                "
            id = str(i)
            plt.title(title + id)
            fig2.canvas.draw()
            plt.savefig((os.getcwd() + "/figures" + "/heatmap_range_theta_v1/" + str(i) + ".png").replace("\\", "/"))
            plt.close("all")





            # RANGE PROFİLE dB
            fig3, ax3 = plt.subplots()
            title = "Range Profile                                                                      "
            id = str(i)
            ax3.set(xlabel='Range axis (meters)', ylabel='Relative Power (dB)', title=(title + id))
            ax3.grid()
            ax3.plot(rangeAxiss, Range_Profile_Average)
            plt.savefig((os.getcwd() + "/figures" + "/range_profile/" + str(i) + ".png").replace("\\", "/"))
            plt.close("all")





            # HEAT MAP on RANGE-THETA AXİS (Heatmap version-3)
            rlist = rangeArrayy
            Angle_Grid = np.array(thetaa)
            Angle_Grid = Angle_Grid + 90.0
            thetalist = np.radians(Angle_Grid)
            rmesh, thetamesh = np.meshgrid(rlist, thetalist)
            fig4, ax4 = plt.subplots(dpi=120, subplot_kw=dict(projection='polar'))
            thetamesh = np.transpose(thetamesh)
            rmesh = np.transpose(rmesh)
            QQ_Average = np.fliplr(QQ_Average)

            if not(self.ui.pushButton_10.isChecked()):
                if (colorbar_min == 0) & (colorbar_max == 0):
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, self.mag2db(QQ_Average), 100, cmap='jet')
                elif (colorbar_min == 0) & (colorbar_max != 0):
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, self.mag2db(QQ_Average), 100, cmap='jet',vmax=colorbar_max)
                elif (colorbar_min != 0) & (colorbar_max == 0):
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, self.mag2db(QQ_Average), 100, cmap='jet',vmin=colorbar_min)
                else:
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, self.mag2db(QQ_Average), 100, cmap='jet',vmin=colorbar_min, vmax=colorbar_max)

                plt.text(7.1, 10.0,"Max dB : " + str(format(np.max(self.mag2db(QQ_Average)), '.3f')) + "\n" + "Min  dB : " + str(format(np.min(self.mag2db(QQ_Average)), '.3f')))
                cbar4 = fig4.colorbar(Heat_Map4)
                cbar4.set_label('Desibel (dB)')
            else:
                if (colorbar_min == 0) & (colorbar_max == 0):
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, QQ_Average, 100, cmap='jet')
                elif (colorbar_min == 0) & (colorbar_max != 0):
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, QQ_Average, 100, cmap='jet',vmax=colorbar_max)
                elif (colorbar_min != 0) & (colorbar_max == 0):
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, QQ_Average, 100, cmap='jet',vmin=colorbar_min)
                else:
                    Heat_Map4 = ax4.contourf(thetamesh, rmesh, QQ_Average, 100, cmap='jet',vmin=colorbar_min, vmax=colorbar_max)

                plt.text(7.1, 10.0,"Max Mag: " + str(format(np.max((QQ_Average)), '.3f')) + "\n" + "Min Mag : " + str(format(np.min((QQ_Average)), '.3f')))
                cbar4 = fig4.colorbar(Heat_Map4)
                cbar4.set_label('Magnitude (Mag)')

            title = "Range/Azimuth Heat Map          "
            id = str(i)
            plt.title(title + id)
            plt.savefig((os.getcwd() + "/figures" + "/heatmap_range_theta_v2/" + str(i) + ".png").replace("\\", "/"))
            plt.close("all")



            # keep max db or mag for each dat file(measurement)
            if not(self.ui.pushButton_10.isChecked()):
                max_db.append(float(format(np.max(self.mag2db(QQ_Average)), '.3f')))
                max_db_axis.append(str(i))
            else:
                max_db.append(float(format(np.max((QQ_Average)), '.3f')))
                max_db_axis.append(str(i))





        # Max dB or Mag VİEW
        fig5, ax5 = plt.subplots()

        if not (self.ui.pushButton_10.isChecked()):
            ax5.set(xlabel='Measurements', ylabel='Max dB', title="Max dB of Measurements")
        else:
            ax5.set(xlabel='Measurements', ylabel='Max Mag', title="Max Mag of Measurements")

        ax5.plot(max_db_axis, max_db, 'o-b')
        ax5.grid()

        if not(self.ui.pushButton_10.isChecked()):
            plt.ylim(50, 150)
            plt.yticks(np.linspace(50, 150, 20))  # plt.yticks(np.linspace(np.min(max_db),np.max(max_db),20))
        else:
            plt.ylim((np.min(max_db)), (np.max(max_db)))
            plt.yticks(np.linspace((np.min(max_db)),(np.max(max_db)),20))

        plt.savefig((os.getcwd() + "/figures" + "/max_view.png").replace("\\", "/"))
        plt.close("all")

        self.ui_cfg.ui.listWidget.addItem("------------------------------------------------------------------Finish-----------------------------------------------------------------------")

        # If you want to see the graphics on the console
        # plt.show()



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())

