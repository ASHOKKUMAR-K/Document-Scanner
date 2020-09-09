<h1 align="center">Document Scanner</h1>

## Sample Input / Output
![sample input output](output/final-output.png)

## Process Taken to Scan Image

- Converting JPG image to RGB matrix format.
- Resizing RGB Image Matrix to 640 x 480 pixels.
    - Group of pixels are replaced by a single pixel with their mean value.
- Converting RGB Image Matrix to Grayscale Image matrix
    - Gray = (Red + Green + Blue) / 3
    - References: [Grayscale to RGB Conversion - Tutorialspoint](https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm#:~:text=Since%20its%20an%20RGB%20image,get%20your%20desired%20grayscale%20image.&text=If%20you%20have%20an%20color,into%20grayscale%20using%20average%20method.)
- Giving 1px black border on all four sides for calculation process.
