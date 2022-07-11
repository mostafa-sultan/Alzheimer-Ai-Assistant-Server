const nodeWebCam = require('node-webcam');
const fs = require('fs');
const express = require('express');
const app = express();
const path = require('path');
var axios = require('axios');
var FormData = require('form-data');
app.use(express.static('images')) // images folder to be served
// Now we can just say localhost:3000/image.jpg

// specifying parameters for the pictures to be taken
var options = {
    width: 1280,
    height: 720,
    quality: 100,
    delay: 1,
    saveShots: true,
    output: "jpg",
    device: false,
    callbackReturn: "location"
};

// create instance using the above options
var webcam = nodeWebCam.create(options);

// capture function that snaps <amount> images and saves them with the given name in a folder of the same name
var captureShot =  (amount, i, name) => {
    // Make sure this returns a real url to an image.
    return new Promise(resolve => {
        var path = `./images/${name}`;

        // create folder if and only if it does not exist
        if (!fs.existsSync(path)) {
            fs.mkdirSync(path);
        }

        // capture the image
        webcam.capture(`./images/${name}/${name}${i}.${options.output}`, (err, dat) => {

            var data = new FormData();
            data.append('file', fs.createReadStream(dat));
            console.log("1");
            var config = {
                method: 'post',
                url: 'http://127.0.0.1:3000/detectAllDataFace',
                headers: {
                    ...data.getHeaders()
                },
                data: data
            };

             axios(config)
                .then(function (response) {
                    console.log("2");
                    console.log(JSON.stringify(response.data));
                })
                .catch(function (error) {
                    console.log(error);
                });

            if (!err) {
                console.log('Image created')
            }
            console.log(err);
            // i++;
            // if (i <= amount) {
            // captureShot(1, 1, name)
            //     .then((response) => {
            //         lo

            //         // Whatever we resolve in captureShot, that's what response will contain
            //         res.send('<img src="${response}"/>')
            //     });







            // }
            resolve('/path/to/image.jpg')
            console.log("datdat");
            console.log(dat);





            // return data;
        });
    })

};

// call the capture function


app.get('/',async (req, res) => {
    await captureShot(1, 1, 'robin')
        .then((response) => {
            console.log("Hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh");
            console.log(response);
        //     var data = new FormData();
        //     data.append('file', response);

        //     var config = {
        //         method: 'post',
        //         url: 'http://127.0.0.1:3000/detectAllDataFace',
        //         headers: {
        //             ...data.getHeaders()
        //         },
        //         data: data
        //     };

        //     axios(config)
        //         .then(function (response) {
        //             console.log(JSON.stringify(response.data));
        //         })
        //         .catch(function (error) {
        //             console.log(error);
        //         });
        //     //     // Whatever we resolve in captureShot, that's what response will contain
        //     res.send('<img src="${response}"/>')
            })
        });

    app.listen(3001, () => {
        console.log("Listening at port 3001....");
    });