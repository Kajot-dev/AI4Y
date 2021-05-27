const https = require("https");
var lib = {};
lib.url = "https://thispersondoesnotexist.com/image";
lib.craftRequest = function(stream, callback, trynum) {
  trynum = typeof trynum === "number" ? trynum : 1;
  https.get(lib.url, (res) => {
    if (res.statusCode == 200) {
      res.pipe(stream);
      res.on("end", () => {
        callback();
      });
    } else {
      console.error("Server responded with code: "+res.statusCode);
      if (trynum < 3) {
        console.log("Trying again in 5 seconds...");
        setTimeout(() => {
          lib.craftRequest(stream, callback, trynum+1);
        }, 5000);
      } else {
        console.error("Error occured too many times! Exiting...");
        process.exit(1);
      }
    }
  });
};
module.exports = lib;