#Author: Maximilian Beckers, EMBL Heidelberg, Sachse Group (2018)

import numpy as np
import time
import argparse, os, sys
import mrcfile
import math
from optimizationUtil import *
from FDRutil import *

#*************************************************************
#****************** Commandline input ************************
#*************************************************************
 
cmdl_parser = argparse.ArgumentParser(
	prog=sys.argv[0], description='*** Analyse density ***', 
	formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=True);

cmdl_parser.add_argument('-em', '--em_map', default=[], nargs='*', required=True, help='Input filename of EM-frame maps');
cmdl_parser.add_argument('-p', '--apix', metavar="apix",  type=float, required=True, help='pixel Size of input map');
cmdl_parser.add_argument('-lowPassFilter', '--lowPassFilter', type=float, required=False,  help='Resolution to lowPass filter');
cmdl_parser.add_argument('-addFrames', '--addFrames', type=int, required=False, help='add Frames');

#--------------------------------------------------------------------------
def kernelRegression(frameData, providedResolution):

	#*****************************************
	#*********** kernel regression ***********
	#*****************************************

	bandwidth = 3;

	maps = np.copy(frameData);
	sizeMap = maps.shape;
	numFrames = sizeMap[3];

	#if specified, filter all the frames to make fallof estimation more accurate
	if providedResolution != 0:
		frequencyMap = calculate_frequency_map(maps[ :, :, :, 0]);
		for frameInd in range(sizeMap[3]):
			maps[:, :, :, frameInd] = lowPassFilter(np.fft.rfftn(maps[:, :, :, frameInd]), frequencyMap, providedResolution, maps[ :, :, :, frameInd].shape);

	#set all negative values to a very small positive value
	maps[maps <= 0.0] = 1.0*10**(-6);

	#do log-transform of maps to linearize data
	#maps = np.log(maps);

	indexMap = np.zeros(sizeMap[3]);
	for i in range(sizeMap[3]):
		indexMap[i] = i+1.0;

	x_mean = np.mean(indexMap);
	y_mean = np.mean(maps, 3);

	regrMap = np.zeros(sizeMap);

	#do the actual kernel regression
	for frameInd in range(numFrames):
		regrMapDenom = 0;
		for tmpFrameInd in range(numFrames):
			dist = np.abs(tmpFrameInd - frameInd);
			
			if dist > 4:
				continue;
			
			sampleWeight = (1.0/(np.sqrt(2*np.pi)*bandwidth)) * np.exp(-0.5*dist/(bandwidth**2));
			regrMap[ :, :, :, frameInd] = regrMap[ :, :, :, frameInd] + sampleWeight*maps[ :, :, :, tmpFrameInd]  ; 
			regrMapDenom = regrMapDenom + sampleWeight; 

		regrMap[ :, :, :, frameInd] = regrMap[ :, :, :, frameInd]/regrMapDenom;

	#************************************
	#*********** do plotting ************
	#************************************
	
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(5, 5);

	for row in ax:
		for col in row:
			
			xInd = np.random.randint(0, sizeMap[0]);
			yInd = np.random.randint(0, sizeMap[1]);
			zInd = np.random.randint(0, sizeMap[2]);

			indices = np.arange(sizeMap[3]);
			y1 = regrMap[ xInd, yInd, zInd, :];
			y2 = maps[ xInd, yInd, zInd, :];
			
			col.plot(indices, y1);
			col.plot(indices, y2);
			col.set_xticklabels([]);
			col.set_yticklabels([]);

	plt.savefig("Regression_falloff.pdf");
	print("PDF saved ...");
	plt.close();

	#calculate weights
	weightMap = np.copy(regrMap);
	sumMap = np.sum(regrMap, 3); 
	sumMap = sumMap.astype(float);
	sumMap[sumMap==0.0] = np.nan;
	
	for frameInd in range(sizeMap[3]):
		weightMap[:, :, :, frameInd] = weightMap[:, :, :, frameInd]/sumMap;
		weightMap[np.isnan(weightMap)] = 1.0/numFrames;

	return regrMap, weightMap;

#--------------------------------------------------------------------------
def linearizedModel(frameData, providedResolution):

	#****************************************
	#*********** fit linear model ***********
	#****************************************
	maps = np.copy(frameData);
	sizeMap = maps.shape;

	#print(sizeMap);
	#if specified, filter all the frames to make fallof estimation more accurate
	if providedResolution != 0:
		frequencyMap = calculate_frequency_map(maps[ :, :, :, 0]);
		for frameInd in range(sizeMap[3]):
			maps[:, :, :, frameInd] = lowPassFilter(np.fft.rfftn(maps[:, :, :, frameInd]), frequencyMap, providedResolution, maps[ :, :, :, frameInd].shape);

	#set all negative values to a very small positive value
	maps[maps<= 0.0] = 1.0*10**(-6);

	#do log-transform of maps to linearize data
	maps = np.log(maps);

	indexMap = np.zeros(sizeMap[3]);
	for i in range(sizeMap[3]):
		indexMap[i] = i+1.0;

	x_mean = np.mean(indexMap);
	y_mean = np.mean(maps, 3);

 	#calc b1
	S_xy = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	S_xx = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	#S_yy = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));

	for i in range(sizeMap[3]):
		S_xy = S_xy + (indexMap[i] - x_mean)*(maps[ :, :, :, i ] - y_mean); 
		S_xx = S_xx + (indexMap[i] - x_mean)**2;
		#S_yy = S_yy + (maps[ :, :, :, i ] - y_mean)*(maps[ :, :, :, i ] - y_mean);

	#calculate regression coefficients
	b1 = np.divide(S_xy, S_xx);
	b0 = y_mean - b1 * x_mean;
	
	#calculate some residual statistics
	#S_residuals = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	#for frameInd in range(sizeMap[3]):
	#	prediction = b0 + b1*(frameInd + 1.0);
	#	S_residuals = S_residuals + (maps[ :, :, :, i ] - prediction)**2;

	#S_residuals[S_residuals == 0] = np.nan;
	#calculate t-test upon b1, H_0: b1=0, H1: b1<0
	#tTestMap = b1/(np.sqrt(S_residuals/(float(sizeMap[3]-2.0))*S_xx));
	
	#np.random.shuffle(b1);

	return b0, b1;

#--------------------------------------------------------------------------
def relativeSNR(weightMap, apix):
	
	sizeMap = weightMap.shape;
	equalWeightFactor = 1.0/float(sizeMap[3]);
	
	S_xq = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	S_xx = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	S_yy = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));

	for frameInd in range(sizeMap[3]):
		S_xq = S_xq + weightMap[:,:,:, frameInd]*equalWeightFactor;
		S_xx = S_xx + equalWeightFactor**2;
		S_yy = S_yy + weightMap[:,:,:, frameInd]**2;

	SNRmap = np.divide((np.sqrt(S_xx)*np.sqrt(S_yy)), S_xq);
	
	#write the SNR map
	SNRMapMRC = mrcfile.new('SNR.mrc', overwrite=True);
	SNRmap = np.float32(SNRmap);
	SNRMapMRC.set_data(SNRmap);
	SNRMapMRC.voxel_size = apix;
	SNRMapMRC.close();

	return None;

#--------------------------------------------------------------------------
def weightedAverage(maps, weightMap):

	indexMap = np.copy(maps);
	sizeMap = maps.shape;

	#do the weighted averaging
	weightedAverage = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));	
	for frame in range(sizeMap[3]):
		weightedAverage = weightedAverage + weightMap[ :, :, :, frame] * maps[ :, :, :, frame];

	#variance map
	#varMap = np.sum(weightMap**2 , 3);
	#weightedAverage = np.divide(weightedAverage, np.sqrt(varMap)); #normalize the background variance

	return weightedAverage;

#--------------------------------------------------------------------------
def optimizeWeights(fallOff):

	numFrames = fallOff.size;
	
	#get starting point
	alphaStart = fallOff/np.sum(fallOff);
	#alphaStart = np.random.rand(numFrames)/np.sum(fallOff);

	x = alphaStart[1:];
	x = gradientDescent(x, fallOff);

	#transform back to alpha
	alpha = np.append(np.ones(1), x);
	basisTransform = np.identity(alpha.size);
	basisTransform[0,:] = -1.0 * np.ones((alpha.size));
	basisTransform[0,0] = 1.0;
	
	#transform into the n-dimensional space
	alphaFinal = np.matmul(basisTransform, alpha);

	return alphaFinal;

#-------------------------------------------------------------------------
def calculateMask(maps):
	
	meanMap = np.mean(maps,3);

	mask = np.copy(meanMap);
	mask[mask>0.0002] = 1;
	mask[mask<1] = 0;

	return mask;

#--------------------------------------------------------------------------
def calcR2(frameData, b0, b1, sizeMovie):
	
	maps = np.copy(frameData);
	#set all negative values to a very small positive value
	maps[maps<= 0.0] = 1.0*10**(-6);

	#do log-transform of maps to linearize data
	maps = np.log(maps);
		
	yMean = np.mean(maps, 3);

	weightMap = np.zeros(sizeMovie);
	#set all increasing fits to zero
	b1[b1>0.0] = 0.0;	
	b0[b1==0.0] = yMean[b1==0]; 

	#get falloff factor for all frames
	for frameInd in range(sizeMovie[3]):
		#weightMap[ :, :, :, frameInd] = b0 + (frameInd+1)*b1;
		weightMap[ :, :, :, frameInd] = N0*np.exp((frameInd+1)*b1);		
	
	#get R2
	weightMean = np.mean(weightMap, 3);
	
	SQE = np.zeros((sizeMovie[0], sizeMovie[1], sizeMovie[2]));
	SQT = np.zeros((sizeMovie[0], sizeMovie[1], sizeMovie[2]));
	
	for frameInd in range(sizeMovie[3]):
		SQE = SQE + (weightMap[:,:,:,frameInd] - weightMean)**2;	
		SQT = SQT + (maps[:,:,:,frameInd] - yMean)**2;

	SQT[SQT==0.0] = np.nan;
	R2 = np.divide(SQE,SQT);
	R2[np.isnan(R2)] == 0;

	return R2;

#--------------------------------------------------------------------------
def calculateWeights(b0, b1, sizeMovie, frameData, firstPatch):
	
	maps = np.copy(frameData);
	#maps[maps<= 0.0] = 1.0*10**(-6);
	#maps = np.log(maps);
	
	yMean = np.mean(maps, 3);

	weightMap = np.zeros(sizeMovie);

	#set all increasing fits to zero
	b1[b1>0.0] = 0.0;	

	#b0[b1==0.0] = yMean[b1==0]
	b0[b1==0.0] = np.nan; 
	N0 = np.exp(b0);
	N0[np.isnan(N0)] = 0.0;

	#get falloff factor for all frames
	for frameInd in range(sizeMovie[3]):
		#weightMap[ :, :, :, frameInd] = b0 + (frameInd+1)*b1;
		weightMap[ :, :, :, frameInd] = N0*np.exp((frameInd+1)*b1);		

	#************************************
	#*********** do plotting ************
	#************************************
	
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(5, 5);

	#index = np.array(np.nonzero(mask));
	#sizeIndex= index.shape;
	for row in ax:
		for col in row:
			
			#voxel = np.random.randint(0, sizeIndex[1]); 
			
			xInd = np.random.randint(0, sizeMovie[0]);
			yInd = np.random.randint(0, sizeMovie[1]);
			zInd = np.random.randint(0, sizeMovie[2]);

			indices = np.arange(0,sizeMovie[3]);
			y1 = weightMap[ xInd, yInd, zInd, :];
			y2 = maps[ xInd, yInd, zInd, :];
			
			col.plot(indices, y1);
			col.plot(indices, y2);
			col.set_xticklabels([]);
			col.set_yticklabels([]);

	plt.savefig("Regression_falloff.pdf");
	plt.close();

	#***********************************
	#(xArr, yArr, zArr) = np.nonzero(testMap); #get indices of nonzero elements
	#for frameInd in range(sizeMovie[3]):
	#	weightMap = weightMap[:, :, :, frameInd] + np.amin(weightMap, 3);
	
	sumMap = np.sum(weightMap, 3);
	sumMap = sumMap.astype(float);
	sumMap[sumMap==0.0] = np.nan;
	for frameInd in range(sizeMovie[3]):
		weightMap[:, :, :, frameInd] = weightMap[:, :, :, frameInd]/sumMap;
		weightMap[np.isnan(weightMap)] = 1.0/float(sizeMovie[3]);	
	
	return weightMap;

#--------------------------------------------------------------------------
def leastSquaresLoss(maps, lambdas, N0):

	#*******************************************
	#**** least squares with exp. decay *******
	#*******************************************

	sizeMap = maps.shape;
	indMaps = np.zeros(sizeMap);
	numFrames = sizeMap[3];

	#get response map
	unscaledResponseMap = np.zeros(sizeMap);
	sumSqDiffs = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	d_lambda = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	d_N0 = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));

	for frameInd in range(numFrames):

		unscaledResponseMap[ :, :, :, frameInd] = np.exp(-lambdas*(frameInd+1));
		responseMap = N0 * unscaledResponseMap[ :, :, :, frameInd];
		diff = responseMap[:,:,:] - maps[:,:,:,frameInd];

		#sumSquaredDifferences
		sumSqDiffs = sumSqDiffs + diff*diff;

		#gradients
		d_lambda = d_lambda + 2.0 * diff * responseMap[ :, :, :] * (-frameInd);
		d_N0 = d_N0 + 2.0 * diff * responseMap[ :, :, :] * unscaledResponseMap[ :, :, :, frameInd];

	return sumSqDiffs, d_lambda, d_N0;

#-------------------------------------------------------------------------
def getInitialMapStats(fileNameList):
	
	tmpMap = mrcfile.open(fileNameList[0], mode='r+');
	tmpMapData = np.copy(tmpMap.data);

	#get the map stats
	sizeMap = tmpMapData.shape;
	patchSize = 200;
	numXPatches = int(math.ceil((sizeMap[0])/float(patchSize)));
	numYPatches = int(math.ceil((sizeMap[1])/float(patchSize)));
	numZPatches = int(math.ceil((sizeMap[2])/float(patchSize)));

	return numXPatches, numYPatches, numZPatches, patchSize, sizeMap;

#-------------------------------------------------------------------------
def main():

	startTime = time.time();
	
	#**********************************
	#**** catch command line input ****
	#**********************************
	args = cmdl_parser.parse_args(); 
	numFrames = len(args.em_map);
	apix = args.apix;

	if args.lowPassFilter is not None:
		lowPassFilter = apix/args.lowPassFilter;
	else:
		lowPassFilter = 0;

	if args.addFrames is None:
		addFrames = 1;
	else:
		addFrames = args.addFrames;
	
	#some initialization
	numXPatches, numYPatches, numZPatches, patchSize, sizeMap = getInitialMapStats(args.em_map);
	splitFilename = os.path.splitext(os.path.basename(args.em_map[0]));
	weightedMap = np.zeros(sizeMap);
	b0Map = np.zeros((sizeMap[0], sizeMap[1], sizeMap[2]));
	b1Map = np.zeros(sizeMap);

	#*********************************
	#*** print the filenames *********
	#*********************************
	
	print("Printing file names. Make sure they are in the correct order ...");
	for filename in args.em_map:
		print(filename);
	
	#*********************************
	#******* do the weighting ********
	#*********************************

	numPatches = 1;
	for xPatchInd in range(numXPatches):
		for yPatchInd in range(numYPatches):
			for zPatchInd in range(numZPatches):
				
				output = "Analyzing patch " + repr(numPatches) + " ...";
				print(output);
				
				if numPatches == 1:
					firstPatch = True;
				else:
					firstPatch = False;

				#**********************************
				#********* read the maps **********
				#**********************************

				#define start and end indices for subsetting
				xStart = patchSize*xPatchInd;
				xEnd = np.minimum(patchSize*(xPatchInd+1), sizeMap[0]);

				yStart = patchSize*yPatchInd;
				yEnd = np.minimum(patchSize*(yPatchInd+1), sizeMap[1]);

				zStart = patchSize*zPatchInd;
				zEnd = np.minimum(patchSize*(zPatchInd+1), sizeMap[2]);

				#read the individual frames
				frameInd = 0;
				addFrameInd = 0; #for adding subsequent frames
				allFrameInd = 0;
				for filename in args.em_map:
					tmpMap = mrcfile.open(filename, mode='r+');
					tmpMapData = np.copy(tmpMap.data);
					
					mapPatch = tmpMapData[xStart:xEnd, yStart:yEnd, zStart:zEnd];
					
					if frameInd == 0:
						tmpSizeMap = mapPatch.shape;
						newNumFrames = int(math.ceil(numFrames/float(addFrames)));
						maps = np.zeros((tmpSizeMap[0], tmpSizeMap[1], tmpSizeMap[2], newNumFrames));
						tmpMapPatch = np.zeros(tmpSizeMap);
						#print(maps.shape);

					if addFrames == 1: #if no frame name reduction takes place
							
						maps[ :, :, :, frameInd] = mapPatch;
						mapPatch = 0;
						frameInd = frameInd + 1;
				
					else: #if subsequent frames shall be added
						
						addFrameInd = addFrameInd + 1;
						allFrameInd = allFrameInd + 1;
						tmpMapPatch = tmpMapPatch + mapPatch; 
							
						if addFrameInd == addFrames:
							tmpMapPatch = (1.0/float(addFrames))*tmpMapPatch;	
							maps[ :, :, :, frameInd] = np.copy(tmpMapPatch);
							tmpMapPatch = np.zeros(tmpSizeMap);			
							
							mapPatch = 0;	
							frameInd = frameInd + 1;
							addFrameInd = 0;
							continue;
						
						if	allFrameInd == numFrames: #if some frames remain after frame reduction add them as well
							remainingNumberOfFrames = numFrames%addFrames;
							tmpMapPatch = (1.0/float(remainingNumberOfFrames))*tmpMapPatch;
							maps[ :, :, :, frameInd] = np.copy(tmpMapPatch);
							#print("assigning last part");

				#**********************************
				#******** calc averages ***********
				#**********************************

				b0, b1 = linearizedModel(maps, lowPassFilter);
				#R2 = calcR2(maps, b0, b1, maps.shape);
				#mask = calculateMask(maps);
				#mask = np.ones(mask.shape);
				weightMap = calculateWeights(b0, b1, maps.shape, maps, firstPatch);
				#weightMap = np.ones(weightMap.shape)*1.0/float(numFrames);
				#regrMap, weightMap = kernelRegression(maps, lowPassFilter);
				weightedSum = weightedAverage(maps, weightMap);	
				b0Map[xStart:xEnd, yStart:yEnd, zStart:zEnd] = b0;
				b1Map[xStart:xEnd, yStart:yEnd, zStart:zEnd] = b1;

				weightedMap[xStart:xEnd, yStart:yEnd, zStart:zEnd] = weightedSum;
	
				if numPatches == 1:
					relativeSNR(weightMap, apix); #write the SNR map

				numPatches = numPatches + 1; 
	
	#end of weighting

	#write output
	weightedMapMRC = mrcfile.new(splitFilename[0] + '_DW.mrc', overwrite=True);
	weightedMap = np.float32(weightedMap);
	weightedMapMRC.set_data(weightedMap);
	weightedMapMRC.voxel_size = apix;
	weightedMapMRC.close();

	#write b0 and b1
	b0MRC = mrcfile.new(splitFilename[0] + '_b0.mrc', overwrite=True);
	b0Map = np.float32(b0Map);
	b0MRC.set_data(b0Map);
	b0MRC.voxel_size = apix;
	b0MRC.close();

    #write b0 and b1
	b1MRC = mrcfile.new(splitFilename[0] + '_b1.mrc', overwrite=True);
	b1Map = np.float32(b1Map);
	b1MRC.set_data(b1Map);
	b1MRC.voxel_size = apix;
	b1MRC.close();

	endTime = time.time();
	runTime = endTime - startTime;
	output = "Elapsed runtime was " + repr(runTime) + " seconds";
	print(output);

if (__name__ == "__main__"):
	main();



