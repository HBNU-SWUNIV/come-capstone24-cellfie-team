const mongoose = require("mongoose");

exports.mongoDB = () => {
  mongoose
    .connect(
      "mongodb+srv://20191744:Im1Ffiiwmmm4zORp@cluster0.hxbg79g.mongodb.net/"
    )
    .then(() => console.log("connected"))
    .catch(() => console.log("mongodb connection failed"));
};
