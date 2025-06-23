Working towards PIG (Puzzle Image Generation)
- The completed puzzle: a single channel image, like grayscale.
- The puzzle pieces: the distribution of the channel's pixel values. i.e., 205 pixels of value 7, 847 pixels of value 9, ..., 91 pixels of value 223, etc.
- The puzzle guide image: one or more downsampled versions of the original image channel. This could be the result of basic resize, edge detection, binary thresholding, etc.
- The puzzle assembly: generate the completed puzzle based on the downsampled image channel and whatever other guides help in the process of reconstructing the original image.
- Bonus, puzzle verification: run the predicted completed puzzle through a hash or other fuction of some sort, having run the actual completed puzzle through the same function and stored the result as part of the compressed data, and check whether the two results match.
- Oink.
