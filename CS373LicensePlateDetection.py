import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array



# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate5.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here
    
    #conversion to greyscale
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    #contrast stretching
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    #Filtering to detect high contrast regions (computing standard deviation in 5x5 pixel neighbourhood)
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)

    #stretch results between 0 and 255 (contrast stretching)
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    #Thresholding for segmentation (high contrast binary image. note: good threshold = 150)
    px_array = computeThresholdSegmentation(px_array, image_width, image_height, 150)

    #Morphological operations (several 3x3 dilation then several 3x3 erosion to get 'blob' region)
    #diliation
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    #erosion
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    
    #Connected component analysis
    (px_array, IDmapping) = computeConnectedComponentLabeling(px_array, image_width, image_height)

    #find liscese and compute bounding box
    (bbox_min_y, bbox_max_y, bbox_min_x, bbox_max_x) = computeLargestComponentRatio(px_array, image_width, image_height, IDmapping)


    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()

def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for h in range(image_height):
        for w in range (image_width):
            gVal = round(0.299 * pixel_array_r[h][w] + 0.587 * pixel_array_g[h][w] + 0.114 * pixel_array_b[h][w])
            greyscale_pixel_array[h][w] = gVal
    return greyscale_pixel_array

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    fLow = 255
    fHigh = 0
    for i in pixel_array:
        for j in i:
            num = j
            if num < fLow:
                fLow = num
            if num > fHigh:
                fHigh = num
   
    st = 255
    if fHigh - fLow == 0:
        st = 1
    else:
        st= 255/(fHigh - fLow)
    
    for h in range(image_height):
        for w in range(image_width):
            sOut = round((pixel_array[h][w] - fLow) * st)
            if sOut < 0:
                result[h][w] = 0
            elif sOut > 255:
                result[h][w] = 255
            else:
                result[h][w] = sOut
    return result

def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)
    for h in range(2, image_height-2):
        for w in range(2, image_width-2):
            
            output_pixel = 0.0
            sumVal = 0.0
            topMost = pixel_array[h-2][w-2 : w+3]
            top = pixel_array[h-1][w-2 : w+3]
            middle = pixel_array[h][w-2 : w+3]
            bottom = pixel_array[h+1][w-2 : w+3]
            bottomMost = pixel_array[h+2][w-2 : w+3]

            joinedVals = topMost + top + middle + bottom + bottomMost
            meanVal = sum(joinedVals)/len(joinedVals)
            for x in joinedVals:
                sumVal += math.pow((x - meanVal),2)
            
            output_pixel = math.sqrt(sumVal/len(joinedVals))
            result[h][w] = output_pixel
    return result

def computeThresholdSegmentation(pixel_array, image_width, image_height, threshold):
    result = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)
    for h in range(image_height):
        for w in range(image_width):
            output_pixel = 0.0
            current_pixel = pixel_array[h][w]
            if current_pixel >= threshold:
                output_pixel = 255
            else:
                output_pixel = 0.0
            
            result[h][w] = output_pixel
    return result

def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    maxLength = image_width + 2
    maxHeight = image_height + 2
    copied_array = createInitializedGreyscalePixelArray(maxLength, maxHeight)
    for y in range(1, maxHeight-1):
        for x in range(1, maxLength-1):
            copied_array[y][x] = pixel_array[y-1][x-1]
    
    for h in range(1, maxHeight-1):
        for w in range(1, maxLength-1):
            if copied_array[h][w] >= 1:
                result = computeDilationNeighbour(h-1, w-1, result, image_height, image_width)
    

    return result
    
def computeDilationNeighbour(h, w, result, height, width):
    if h-1 >= 0 and w-1 >=0:
        result[h-1][w-1] = 1
    if h-1 >= 0 and w >=0:
        result[h-1][w] = 1
    if h-1 >= 0 and w+1 < width:
        result[h-1][w+1] = 1
    if h >= 0 and w-1 >=0:
        result[h][w-1] = 1
    if h >= 0 and w >=0:
        result[h][w] = 1
    if h >= 0 and w+1 < width:
        result[h][w+1] = 1
    if h+1 < height and w-1 >=0:
        result[h+1][w-1] = 1
    if h+1 < height and w >=0:
        result[h+1][w] = 1
    if h+1 < height and w+1 < width:
        result[h+1][w+1] = 1

    return result    



def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    maxLength = image_width + 2
    maxHeight = image_height + 2
    copied_array = createInitializedGreyscalePixelArray(maxLength, maxHeight)
    for y in range(1, maxHeight-1):
        for x in range(1, maxLength-1):
            copied_array[y][x] = pixel_array[y-1][x-1]
    
    for h in range(1, maxHeight-1):
        for w in range(1, maxLength-1):
            if copied_array[h][w] >= 1:
                output_pixel = computeErosionNeighbour(h, w, copied_array)
                result[h-1][w-1] = output_pixel  
    return result
    
def computeErosionNeighbour(h, w, pixel_array):
    topleft = pixel_array[h-1][w-1]
    top = pixel_array[h-1][w]
    topright = pixel_array[h-1][w+1]
    left = pixel_array[h][w-1]
    right = pixel_array[h][w+1]
    bottomleft = pixel_array[h+1][w-1]
    bottom = pixel_array[h+1][w]
    bottomright =  pixel_array[h+1][w+1]
    
    if topleft >= 1 and top >= 1 and topright >= 1 and left >= 1 and right >= 1 and bottomleft >= 1 and bottom >= 1 and bottomright >= 1:
        return 1
    return 0

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    currentLabel=1
    labelCount = 0
    diction = {}
    
    for h in range(image_height):
        for w in range(image_width):
            if pixel_array[h][w] >= 1 and visited[h][w] == 0:
                q = Queue()
                q.enqueue((h,w))
                visited[h][w] = 1

                while q.isEmpty() == False:
                    (y,x) = q.dequeue()
                    result[y][x] = currentLabel
                    labelCount +=1
                    #top
                    if y-1 >= 0:
                        if pixel_array[y-1][x] >= 1 and visited[y-1][x] == 0:
                            q.enqueue((y-1,x))
                            visited[y-1][x] = 1
                    #bottom
                    if y+1 < image_height:
                        if pixel_array[y+1][x] >= 1 and visited[y+1][x] == 0:
                            q.enqueue((y+1,x))
                            visited[y+1][x] = 1                                 
                    #left
                    if x-1 >= 0:
                        if pixel_array[y][x-1] >= 1 and visited[y][x-1] == 0:
                            q.enqueue((y,x-1))
                            visited[y][x-1] = 1
                    #right
                    if x+1 < image_width:
                        if pixel_array[y][x+1] >= 1 and visited[y][x+1] == 0:
                            q.enqueue((y,x+1))
                            visited[y][x+1] = 1
                            
                diction[currentLabel] = labelCount
                labelCount=0
                currentLabel +=1
    return (result, diction)
                
def computeLargestComponentRatio(pixel_array, image_width, image_height, diction):
    sorted_map = sorted(diction.items(), key=lambda x: x[1], reverse=True)
    for key in sorted_map:
        minHeight = image_height
        maxHeight = 0.0
        minWidth = image_width
        maxWidth = 0.0

        for h in range(image_height):
            for w in range(image_width):
                if pixel_array[h][w] == key[0]:
                    if h < minHeight:
                        minHeight = h
                    if h > maxHeight:
                        maxHeight = h
                    if w < minWidth:
                        minWidth = w
                    if w > maxWidth:
                        maxWidth = w

        diffHeight = maxHeight - minHeight
        diffWidth = maxWidth - minWidth
        ratio = diffWidth/diffHeight
        # check if ratio is within range
        if ratio >= 1.5 and ratio < 5:
            return(minHeight, maxHeight, minWidth, maxWidth)
    return (minHeight, maxHeight, minWidth, maxWidth)

if __name__ == "__main__":
    main()