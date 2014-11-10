
im = imread('image-1.png');
q = evaluate_haar_cascade(im,'haarcascade_frontalface_default.xml');
disp('detections:');
q
