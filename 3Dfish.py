
# Algorithm developed by: Ziv Frankenstein, Ph.D. 

### PseudoCode 

import 

def segmentation():
	# Individual cell nuclei segmentation
	for filename in os.listdir(inputFolder):
	
		indices = np.where(markers > 1)
		coordinates = (indices[0], indices[1])

    return dataSegmentation
		
		
def overlap3D(dataSegmentation):
	# Extract and compare coordinates to assure representation of individual cell nuclei
	# Unique ID for coordinates represent individual cell nuclei
	for layer in NucDapi:
	
		if int(var[markers[int(number.group(2)), int(number.group(1))]]) == int(int(cZ) - 1):

    return dataNoOverlap3D

	
def signalDetection(dataNoOverlap3D):
    # Extract coordinates representation for each signal
	# Extract an individual high intensity coordinate for each signal
	# Comparisons of cell nuclei and signals coordinates to associate signals with their corresponding cell nuclei	
    if signal:
		data = {markers[cY, cX]: {signal.group(1): coor}}

    for key, value in sorted(dA.items(), key=lambda item: (item[0][0])):
		
    return dataSignal
	
			
def vector3D(dataSignal):
    # Calculate all possible 3D vector lengths between different signals coordinates for individual cell nuclei
	# Producing a network for each nuclei in which all nodes (different signals) are connected
	# Multiply comparison of 3D vector lengths 
	# Determine which signals are connected through the 3D vector length for individual cell nuclei
	for psb in d[ksb]:
        
        path = math.sqrt((goldx - int(number.group(1))) ** 2 + (goldy - int(number.group(2))) ** 2 + (goldz - int(number.group(3)) * Zinterval) ** 2)
				
    return dataVector3D

	
def write_dataClass(dataVector3D):
        # For individual cell nuclei : determine number of signals, classify 3D vector lengths into co-localization or break-apart profile
		# Classify data distribution based on nuclei patterns 
        for key, value in sorted(d.items(), key=lambda item: (item[0][0])):
		
            if xTemp:
               if len(xTemp) == 2 and xTemp[len(xTemp) - 1] < 1.2 and xgold[0] == 2 and xsb[0] == 2:
				
				
def main():
	# 3D information of Z-stack images
    # Store information in data structure: Nuclei ID and coordinates, FITC and TRITC coordinates with intensities and 3D paths
    dataSegmentation = segmentation()
    dataNoOverlap3D = overlap3D(dataSegmentation)
    dataSignal = signalDetection(dataNoOverlap3D)
    dataVector3D = vector3D(dataSignal)
    write_dataClass(dataVector3D)

if __name__ == "__main__":
    main()
	
	
	
