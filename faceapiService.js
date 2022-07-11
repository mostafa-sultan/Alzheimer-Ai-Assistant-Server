const save = require("./utils/saveFile");
const path = require("path"); 
const tf = require("@tensorflow/tfjs-node"); 
const canvas = require("canvas");
const faceapi = require("@vladmandic/face-api/dist/face-api.node.js");

const modelPathRoot = "./models";


const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

//photo witch compare
const REFERENCE_IMAGE = 'https://m.media-amazon.com/images/M/MV5BMDRhZDBlYzUtZjcxZS00YjA3LTg0NDYtYzJmMjNkMzkzM2FkXkEyXkFqcGdeQWRvb2xpbmhk._V1_.jpg'
const QUERY_IMAGE = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/examples/images/bbt2.jpg'



//mode face detection
let optionsSSDMobileNet;

//function prepare image file
async function image(file) {
  const decoded = tf.node.decodeImage(file);
  const casted = decoded.toFloat();
  const result = casted.expandDims(0);
  decoded.dispose();
  casted.dispose();
  return result;
}

//function get image flie witch prebare and detect All Faces with Face Expressions with Age And Gender
async function detect(tensor) {
  const result = await faceapi.detectAllFaces(tensor).withFaceExpressions().withAgeAndGender();
  return result;
}
 


//main function
async function main(file, filename) {

//brepare face api 
  console.log("FaceAPI single-process test"); 
  await faceapi.tf.setBackend("tensorflow");
  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready(); 
  console.log(
    `Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${
      faceapi.version.faceapi
    } Backend: ${faceapi.tf?.getBackend()}`
  );

  //lode model from model file
  console.log("Loading FaceAPI models");
  const modelPath = path.join(__dirname, modelPathRoot); 
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceExpressionNet.loadFromDisk(modelPath);
  await faceapi.nets.ageGenderNet.loadFromDisk(modelPath); 
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath)


  //option in face dediction
  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
    minConfidence: 0.5,
  });


  const tensor = await image(file);
  const result = await detect(tensor);


  faceRecognition('https://m.media-amazon.com/images/M/MV5BMDRhZDBlYzUtZjcxZS00YjA3LTg0NDYtYzJmMjNkMzkzM2FkXkEyXkFqcGdeQWRvb2xpbmhk._V1_.jpg',file)
  
// print detected faces data
  console.log("Detected faces:", result);

// drow canvas in face detected and stor it
  const canvasImg = await canvas.loadImage(file);
  const out = await faceapi.createCanvasFromMedia(canvasImg);
  faceapi.draw.drawDetections(out, result);
  save.saveFile(filename, out.toBuffer("image/jpeg"));
  console.log(`done, saved results to ${filename}`); 
  tensor.dispose();

  return result;
}


 // Face recognion //////////////////////////////////////////////////////////////

let labeledFaceDescriptors;

async function faceRecognition(link,imageFile) {  
  // get img we need recognize
  const referenceImage = await canvas.loadImage(imageFile) 
  // lode traning img handler function
  if(!labeledFaceDescriptors){
   labeledFaceDescriptors = await loadLabeledImages();  
  }
  // 
  const faceMatcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors,
    0.6
  );
    // get img with and hight
  const displaySize = { width: referenceImage.width, height: referenceImage.height };
  // detect all face and Descriptors of faces 
  const detections = await faceapi
    .detectAllFaces(referenceImage)
    .withFaceLandmarks()
    .withFaceDescriptors();
    // resize to compare Descriptor
  const resizedDetections = faceapi.resizeResults(
    detections,
    displaySize
  );
  // get best match descriptor
  const results = resizedDetections.map((d) =>
    faceMatcher.findBestMatch(d.descriptor)
  );
  // lope all faces an print all if detected or not
  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box;
    console.log(result.toString());
  });
}  

 
// function handel traninig image for tran  model
function loadLabeledImages() { 
  const labels = [
    'Black Widow',
    'Captain America',
    'Captain Marvel',
    'Hawkeye',
    'Jim Rhodes',
    'Thor',
    'Tony Stark'
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) { 
        const img =await canvas.loadImage(
          `https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`
        );
        console.log("kjhlkjkkkkkkkkkkkkkkkkkkkkkkkk");
        console.log(img);
        console.log(descriptions);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
 
module.exports = {
  detect: main,loadLabeledImages
};

 