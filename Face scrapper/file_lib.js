const fs = require("fs");
const path = require("path");
var lib = {};
lib.basePath = path.join(__dirname, "./faces");
lib.initCheck = function() {
  try {
    fs.accessSync("./faces");
  } catch (e) {
    fs.mkdirSync("./faces");
  }
};
lib.getLastNumber = function(callback) {
  var entries = fs.readdirSync(path.join(__dirname, "./faces"), { "withFileTypes": true});
  var names = [];
  for (let entry of entries) {
    if (entry.isFile()) {
      let dot = entry.name.lastIndexOf(".");
      let ext = entry.name.slice(dot+1);
      if (ext.toLowerCase() == "jpg" || ext.toLowerCase() == "jpeg") {
        let name = entry.name.slice(0, dot);
        var n_name = -1;
        try {
          n_name = parseInt(name);
        } catch (e) {
          n_name = -1;
        }
        if (n_name >= 0) {
          names.push(n_name);
        }
      }  
    }
  }
  names.sort((a, b) => {
    if (a < b) return -1;
    if (a >= b) return 1;
  });
  var last;
  if (names.length > 0) {
    last = names.pop();
  } else {
    last = 0;
  }
  process.env.num = last;
  callback(last);
};
lib.getWriteStream = function(callback) {
  let nr = parseInt(process.env.num);
  let n_path = path.join(lib.basePath, `./${nr}.jpg`);
  if (typeof nr == "number") {
    nr++;
    let stream = fs.createWriteStream(n_path);
    process.env.num = nr;
    callback(stream);
  } else {
    console.error("Last number is not defined!");
    process.exit(1);
  }
};
lib.findDuplicates = function(callback) {
  lib.getNames(entries => {
    var basePath = path.join(__dirname, "./faces");
    var deleteMark = [];
    var d_num = 0;
    for (let i = 1; i < entries.length;i++) {
      let stats_1 = fs.statSync(path.join(basePath, `./${entries[i-1]}.jpg`));
      let stats_2 = fs.statSync(path.join(basePath, `./${entries[i]}.jpg`));
      //console.log(stats_1.size, stats_2.size, stats_1.size == stats_2.size);
      if (stats_1.size == stats_2.size) {
        deleteMark.push(entries[i]);
        //i++;
        d_num++;
      }
    }
    callback(d_num, deleteMark);
  });
};
lib.deleteImg = function(list) {
  for (let elem of list) {
    fs.unlinkSync(path.join(__dirname, `./faces/${elem}.jpg`));
  }
  return list.lenght;
};
lib.getNames = function(callback) {
  var entries = fs.readdirSync(path.join(__dirname, "./faces"), { "withFileTypes": true});
  var toDelete = [];
  for (let i = 0; i < entries.length; i++) {
    let ext = entries[i].name.slice(entries[i].name.lastIndexOf(".")+1).toLowerCase();
    if (!entries[i].isFile() || (ext !== "jpeg" && ext !== "jpg")) {
      toDelete.push(i);
    }
  }
  for (let i of toDelete) {
    entries.splice(i, 1);
  }
  for (let i = 0;i < entries.length;i++) {
    entries[i] = parseInt(entries[i].name);
  }
  entries.sort((a, b) => {
    if (a < b) return -1;
    if (a >= b) return 1;
  });
  //console.log(entries);
  lib.entries = entries;
  callback(entries);
};
lib.renameAll = function(callback) {
  lib.getNames(entries => {
    var x = -1;
    for (let i = 0; i < entries.length; i++) {
      if (entries[i] != i) {
        x = i;
        break;
      }
    }
    if (x != -1) {
      fs.renameSync(path.join(__dirname, `./faces/${entries[x]}.jpg`), path.join(__dirname, `./faces/${x}.jpg`));
      lib.renameAll(() => {
        callback();
      });
    } else callback();
  });
};
module.exports = lib;