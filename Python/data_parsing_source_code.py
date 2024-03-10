#References:                                                                              #
#https://github.com/ibaiGorordo/AWR1642-Read-Data-Python-MMWAVE-SDK-2/tree/heatMapDevelop #
###########################################################################################

#-------------------------------------LIBRARIES------------------------

import sys
import time
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------DEFİNES---------------------------

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
directory = (os.getcwd()+"/dat_files").replace("\\","/")

configFileName = 'xwr16xx_BestRangeResolution_UpdateRate_1.cfg'

#vmin , vmax :  Parameters to be added to the GUI


# ----------------------------FUNCTIONS--------------------------------------


# Magnitude to Desibel Converter
def mag2db(magnitude):
    for i in range(256):
        for j in range(63):
            if magnitude[i,j] == 0:
                magnitude[i,j] = 1

    return 20 * np.log10(magnitude)

#------------------------------------------------

# Desibel to Magnitude Converter
def db2mag(decibel):
    return 10**(decibel/20)

#------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters

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

    print("---------------------CFG-------------------")
    print("numDopplerBins",configParameters["numDopplerBins"])
    print("numRangeBins", configParameters["numRangeBins"])
    print("rangeResolutionMeters", configParameters["rangeResolutionMeters"])
    print("rangeIdxToMeters", configParameters["rangeIdxToMeters"])
    print("dopplerResolutionMps", configParameters["dopplerResolutionMps"])
    print("maxRange", configParameters["maxRange"])
    print("maxVelocity", configParameters["maxVelocity"])
    print("Virtual_Antenna_Azim", configParameters["Virtual_Antenna_Azim"])
    print("numAdcSamples",numAdcSamples)
    print("numAdcSamplesRoundTo2",numAdcSamplesRoundTo2)
    print("Virtual_Antenna_Azim", configParameters["Virtual_Antenna_Azim"])


    return configParameters

#------------------------------------------------


# Funtion to read and parse the incoming data
def readAndParseData16xx(data_2,size_2, configParameters):
    print("---------------------Frame-------------------")
    global byteBuffer, byteBufferLength
    print("byteBuffer : ", byteBuffer)
    print("byteBufferLength : ", byteBufferLength)


    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**25
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
    maxBufferSize = 2**25
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
            word = [1, 2**8, 2**16, 2**24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word) # True

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

            # if incoming data is less than the packet size, exit this function
            if startIdx[0] + totalPacketLen > byteCount:
                print("!!! This Frame has not enough length for reading !!!")
                result = 0
                return  result, dataOK, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8] 
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x') 
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word) 
        print("TotalPacketLen : ",totalPacketLen)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x') 
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word) 
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word) 
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word) 
        print("NumDetectedObj : ",numDetectedObj)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word) 
        print("NumTLVs : ",numTLVs)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word) 
        idX += 4  
    

        idX2 = idX

        # Read the TLV messages
        for tlvIdx in range(numTLVs):

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            print(idX2)

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX2:idX2 + 4], word)
            print("TlvType : ",tlv_type)
            idX2 += 4
            tlv_length = np.matmul(byteBuffer[idX2:idX2 + 4], word)
            idX2 += 4
            print("TlvLen : ",tlv_length)

            idX = idX2
            idX2 += tlv_length



            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                # word array to convert 4 bytes to a 16 bit number
                word = [1, 2**8]
                tlv_numObj = np.matmul(byteBuffer[idX:idX + 2], word)
                print("tlv_NumObj : ",tlv_numObj)
                idX += 2
                tlv_xyzQFormat = 2**np.matmul(byteBuffer[idX:idX + 2], word)
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
                #x = x / tlv_xyzQFormat
                #y = y / tlv_xyzQFormat
                #z = z / tlv_xyzQFormat

                # Store the data in the detObj dictionary
                detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                          "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}

                



            elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                word = [1, 2**8]
                Range_Profile = np.zeros(configParameters["numRangeBins"] , dtype='int16')
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
                #NOT FINISHED, CONTINUE IF NECESSARY


        # Remove already processed data, This process is not used for this algorithm. byteBuffer and byteBufferLength are updated at for loop in main part
        if idX > 0 and dataOK == 1:
            shiftSize = idX

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return result,dataOK, frameNumber, detObj, azimMapObject,startIdx[0],totalPacketLen,Range_Profile,rangeAxis,QQ,theta,rangeArray


#------------------------------------------------

# Funtion to update the data and display in the plot
def update(data_1,size_1):

    dataOk = 0
    global detObj
    global azimMapObject
    x = []
    y = []


    # Read and parse the received data
    result_,dataOk, frameNumber, detObj, azimMapObject,startIdx_,totalPacketLen_,Range_Profile_,rangeAxis_,QQ_,theta_,rangeArray_ = readAndParseData16xx(data_1, size_1, configParameters)


    return result_,dataOk,startIdx_,totalPacketLen_,Range_Profile_,rangeAxis_,QQ_,theta_,rangeArray_,azimMapObject

#------------------------------------------------

# Sort downloaded dat files by creation date
def file_info(directory):
    file_list = []
    file_list_new = []

    for i in os.listdir(directory):
        a = os.stat(os.path.join(directory,i)).st_ctime
        file_list.append(a)

    file_list_sorted = sorted(file_list)

    for i in range(len(file_list_sorted)):
        file_list_new.append(os.listdir(directory)[file_list.index(file_list_sorted[i])])

    return file_list_new








# -------------------------------------------------    MAIN   -----------------------------------------


# More than one input can be entered at once
dat_files_number = list(map(int,input('Enter .dat files number : ').split()))
print(dat_files_number)

file_list = file_info(directory)
print(file_list)

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
    
    # constans
    byteBuffer = np.zeros( 2**25 ,dtype = 'uint8')
    byteBufferLength = 0;

    # find and read the relevant dat file
    fp = open(os.path.join(directory, file_list[(len(file_list) - 1) - (len(file_list) - i)]).replace("\\","/"),'rb')
    readNumBytes = os.path.getsize(os.path.join(directory, file_list[(len(file_list) - 1) - (len(file_list) - i)]).replace("\\","/"))
    print(os.path.join(directory, file_list[(len(file_list) - 1) - (len(file_list) - i)]).replace("\\","/"))
    print("ReadNumBytes : ", readNumBytes)
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
    configParameters = parseConfigFile(configFileName)


    # Parsing loop
    detObj = {}
    azimMapObject = {}
    frameData = {}
    currentIndex = 0
    while (totalBytesParsed < readNumBytes):

        try:
            print("--------------------------------------------")
            # Update the data and check if the data is okay
            result,dataOk,headerStartIndex,totalPacketNumBytes,Range_Profile,rangeAxis,QQ,theta,rangeArray,azimMapObject = update(allBinData[totalBytesParsed::1],readNumBytes-totalBytesParsed)
            totalBytesParsed += (headerStartIndex + totalPacketNumBytes)

            if result == 0:
                break

            if dataOk and result:

                # Store the current frame into frameData
                frameData[currentIndex] = detObj
                currentIndex += 1
                print("currentIndex : ", currentIndex)

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
    Range_Profile_Average = Range_Profile_Total/currentIndex
    QQ_Average = QQ_Total/currentIndex



    # Figure Creations and Saving

    #keep max db for each dat file(measurement)
    max_db.append(float(format(np.max(mag2db(QQ_Average)), '.3f')))
    max_db_axis.append(str(i))

    #HEAT MAP on X-Y AXİS (Heatmap version-1)
    fig1,ax1= plt.subplots()
    Heat_Map1 = plt.contourf(azimMapObjectt["posX"], azimMapObjectt["posY"], mag2db(QQ_Average), 100,cmap="jet")  # vmin=-30,vmax=90
    cbar1 = fig1.colorbar(Heat_Map1)
    cbar1.set_label('Desibel (dB)')
    plt.xlabel("Horizontal, x axis (meters)")
    plt.ylabel("Longitudinal, y axis (meters)")
    title = "Range/Azimuth Heat Map                                "
    id = str(i)
    plt.title(title+id)
    plt.text(6.8,6.7,"Max dB : "+str(format(np.max(mag2db(QQ_Average)),'.3f'))+"\n"+"Min  dB : "+str(format(np.min(mag2db(QQ_Average)),'.3f')))
    plt.savefig((os.getcwd() + "/figures"+"/heatmap_x_y/"+str(i)+".png").replace("\\", "/"))


    #HEAT MAP on RANGE-THETA AXİS (Heatmap version-2)
    fig2,ax2= plt.subplots()
    X,Y = np.meshgrid(azimMapObjectt["theta"], azimMapObjectt["range"])
    Heat_Map2 = plt.contourf(X,Y,mag2db(QQ_Average),100,cmap="jet")
    cbar2 = fig2.colorbar(Heat_Map2)
    cbar2.set_label('Desibel (dB)')
    plt.xlabel("Angle axis (degrees)")
    plt.ylabel("Range axis (meters)")
    title = "Range/Azimuth Heat Map                                "
    id = str(i)
    plt.title(title+id)
    fig2.canvas.draw()
    plt.text(84,6.7,"Max dB : "+str(format(np.max(mag2db(QQ_Average)),'.3f'))+"\n"+"Min  dB : "+str(format(np.min(mag2db(QQ_Average)),'.3f')))
    plt.savefig((os.getcwd() + "/figures" + "/heatmap_range_theta_v1/"+str(i)+".png").replace("\\", "/"))


    # RANGE PROFİLE
    fig3, ax3= plt.subplots()
    title = "Range Profile                                                                      "
    id = str(i)
    ax3.set(xlabel='Range axis (meters)', ylabel='Relative Power (dB)', title=(title+id))
    ax3.grid()
    ax3.plot(rangeAxiss, Range_Profile_Average)
    plt.savefig((os.getcwd() + "/figures" + "/range_profile/" +str(i)+".png").replace("\\", "/"))


    #HEAT MAP on RANGE-THETA AXİS (Heatmap version-3)
    rlist = rangeArrayy
    Angle_Grid = np.array(thetaa)
    Angle_Grid = Angle_Grid+90.0
    thetalist = np.radians(Angle_Grid)
    rmesh,thetamesh = np.meshgrid(rlist,thetalist)
    fig4,ax4 = plt.subplots(dpi=120,subplot_kw = dict(projection='polar'))
    thetamesh = np.transpose(thetamesh)
    rmesh = np.transpose(rmesh)
    QQ_Average = np.fliplr(QQ_Average)
    Heat_Map4 =ax4.contourf(thetamesh,rmesh,mag2db(QQ_Average),100,cmap='jet') 
    cbar4 = fig4.colorbar(Heat_Map4)
    cbar4.set_label('Desibel (dB)')
    title = "Range/Azimuth Heat Map          "
    id = str(i)
    plt.title(title+id)
    plt.text(7.1,10.0,"Max dB : "+str(format(np.max(mag2db(QQ_Average)),'.3f'))+"\n"+"Min  dB : "+str(format(np.min(mag2db(QQ_Average)),'.3f')))
    plt.savefig((os.getcwd() + "/figures" + "/heatmap_range_theta_v2/" +str(i)+".png").replace("\\", "/"))



print(max_db)
print(len(dat_files_number))
print(np.arange(1,len(dat_files_number)+1))

# Max dB VİEW
fig5, ax5= plt.subplots()
ax5.set(xlabel='Measurements', ylabel='Max dB', title="Max dB of Measurements")
ax5.plot(max_db_axis ,max_db,'o-b')
ax5.grid()
plt.ylim(50,150)
plt.yticks(np.linspace(50,150,20)) #plt.yticks(np.linspace(np.min(max_db),np.max(max_db),20))
plt.savefig((os.getcwd() + "/figures" + "/max_view.png").replace("\\", "/"))

print("------------------------------------------------Finish---------------------------------------------------------")

# If you want to see the graphics on the console
#plt.show() 