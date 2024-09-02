
const express = require('express')
const app = express()
var fs = require ('fs');
var multer = require('multer');

const getDirectories = async source =>
  (await fs.promises.readdir(source))

var uploadChoices = multer(
    {dest: './uploads/'},
    ).array('uploadedImages', 3);

var uploadInput = multer(
        {dest: './diss-img-tool-lw/imgs/'},
        ).single('uploadedInput');


// generates promise that runs python script 
let runGen = () => {return new Promise((success, nosuccess) => {

    const { spawn } = require('child_process');
    const pyprog = spawn('python3', ["-u", __dirname + '/diss-img-tool-lw/first_image_generator.py']);
    pyprog.on('close', (_) => {

        success(true);         
    });

    pyprog.stdout.on('data', (data) => {
        // turn on when needed for debug
        console.log(data.toString())
    });

    pyprog.stderr.on('data', (data) => {
        console.log(data.toString())
        nosuccess(false);
    });

})};

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, PUT, POST");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
  });

app.use('/options/', express.static('./uploads/'));


// accessed to run the generator
app.get('/', async (req, res) => {

    try{
        const response = await runGen()
        if(response === false){
            res.status(404).send('something went wrong');
        }else{
            res.status(200).send((await getDirectories('./saved')))
        }

    }catch(err){
        console.log(err)
        res.status(404).send('something went wrong');

    }
})

app.get('/option/:number', (req, res) => {
    const option = req.params.number;
    res.sendFile(`./saved/option${option}.png`, { root: '.' })
})

// accessed to run the generator witha  choice inputted
app.get('/choice/:number', async (req, res) => {
    const choice = req.params.number;
    console.log(`choice: ${choice}`)
    fs.copyFile(`./saved/option${choice}.png`, './diss-img-tool-lw/imgs/style.png', (err) => {
        if (err) throw err;
        console.log('copied successfully');
    });
    
    try{
        const response = await runGen()
        if(response === false){
            res.status(404).send('something went wrong');
        }else{
            res.status(200).send(response)
        }
    }catch(err){
        console.log(err)
    }
})

// accessed for python script to send up generated choices
app.post('/', uploadChoices, (request, respond) => {
    if(request.files){
        for(const file of request.files){
            console.log(file)
            fs.rename(file.path, `${file.destination}${file.originalname}`, (err) => {
                if ( err ) console.log('ERROR: ' + err);
            });
        }
    }
    respond.status(200).send('uploaded');
});

// accessed by client to upload sketch
app.post('/upload/sketch', uploadInput, (request, respond) => {
    if(request.file){
            console.log(request.file)
            if(!(request.file.originalname.endsWith('.png')) && !(request.file.originalname.endsWith('.jpg')) && !(request.file.originalname.endsWith('.jpeg')) ){
                fs.unlink(request.file.path, (err) => {
                    if (err) throw err;
                  }); 
                respond.status(450).send('incorrect image file uploaded');
            }else{
                fs.rename(request.file.path, `${request.file.destination}sketch.png`, (err) => {
                    if ( err ) console.log('ERROR: ' + err);
                });
                respond.status(200).send('uploaded successfully');
            }
        }

});

// accessed by client to upload style
app.post('/upload/style', uploadInput, (request, respond) => {
    if(request.file){
        console.log(request.file)
        if(!request.file.originalname.endsWith('.png') && !(request.file.originalname.endsWith('.jpg')) && !(request.file.originalname.endsWith('.jpeg')) ){
            fs.unlink(request.file.path, (err) => {
                if (err) throw err;
              }); 
            respond.status(450).send('incorrect image file uploaded');
        }else{
            fs.rename(request.file.path, `${request.file.destination}style.png`, (err) => {
                if ( err ) console.log('ERROR: ' + err);
            });
            respond.status(200).send('uploaded successfully');
        }
    }

});

app.listen(8080, () => console.log('Application listening on port 8080!'))