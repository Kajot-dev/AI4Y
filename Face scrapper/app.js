const file_lib = require("./file_lib");
const image_lib = require("./image_lib");
const readline = require('readline');
const exec = require('child_process').exec;
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
var app = {};
app.barLenght = 40;
app.imagesNum = 25;
app.imagesDone = 0;
app.reqestDelay = 230;
app.shutdown = function (callback) {
  exec('shutdown.exe /s /t 00', (error, stdout, stderr) => {
    if (error !== null) {
      console.log(`exec error: ${error}`);
      if (callback) callback();
    } else {

    }
  });
};
app.start = function () {
  file_lib.initCheck();
  rl.question("How many images should be downloaded?\nNumber: ", (answer) => {
    var num = 0;
    try {
      num = parseInt(answer);
    } catch (e) {
      console.error("Answer is not a number!");
      process.exit(1);
    }
    if (num > 0) {
      app.imagesNum = num;
      rl.close();
      console.log("Downloading " + app.imagesNum + " image(s)!");
      app.updateConsole(app.barLenght);
      file_lib.getLastNumber(last => {
        app.getImages(app.imagesNum);
      });
    } else {
      console.error("Number is not positive or it's 0");
      process.exit(1);
    }
  });
};
app.getImages = function (number, curNum) {
  curNum = typeof curNum == "number" ? curNum : 1;
  if (curNum > number) {
    console.log("\nFinished!");
    app.imagesDone = 0;
    app.repairDuplicates();
  } else {
    file_lib.getWriteStream(stream => {
      image_lib.craftRequest(stream, () => {
        app.imagesDone++;
        app.updateConsole(app.barLenght);
        setTimeout(() => {
          app.getImages(number, curNum + 1);
        }, app.reqestDelay);
      });
    });
  }
};
app.repairDuplicates = function () {
  file_lib.findDuplicates((num, list) => {
    console.log("Detected " + num + " duplicate(s)!");
    if (num > 0) {
      console.log("Deleting...");
      file_lib.deleteImg(list);
      app.imagesNum = num;
      console.log("Downloading " + num + " missing images...");
      app.updateConsole(app.barLenght);
      app.getImages(num);
    } else {
      console.log("Repair finished! Renaming...");
      file_lib.renameAll(() => {
        console.log("Exiting...");
        if (process.env.auto_shutdown == "1") {
          console.log("Shutting down in 10 seconds...");
          setTimeout(() => {
            app.shutdown(() => {
              //process.exit(0);
            });
          }, 10000);
        }
        //else process.exit(0);
      });
    }
  });
};
app.updateConsole = function (lenght) {
  readline.clearLine(process.stdout, 0);
  readline.cursorTo(process.stdout, 0);
  process.stdout.write(
    app.imagesDone + " z " + app.imagesNum +
    " [" +
    ("█".repeat(Math.round((app.imagesDone / app.imagesNum) * lenght))) +
    ("-".repeat(Math.round((1 - (app.imagesDone / app.imagesNum)) * lenght))) +
    "] " +
    Math.round(((app.imagesDone / app.imagesNum) * 100) * 100) / 100 +
    "%");
};
try {
  app.start();
} catch (e) {
  console.error(e);
}