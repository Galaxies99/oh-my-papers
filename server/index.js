let express = require("express");
let cors = require("cors");
let app = express(cors());

const PORT = 3000;

let UnixSocket = require("unix-domain-socket");
let us = new UnixSocket("/tmp/test-socket");

us.payloadAsJSON();

app.get("/", (req, res) => {
  const dataToSend = {
    inference: [
      {
        left_context: "ResNet",
        right_context: "is a well-built neural network",
      },
      {
        context: "GoogLeNet",
      },
      {
        context: "yolo",
      },
      {
        left_context: "AlphaPose",
        right_context:
          "is a well-known state-of-the-art pose estimation framework.",
      },
    ],
  };

  us.send(dataToSend, (response) => {
    console.log(response);
  });
  res.send(`Hello!`);
});

app.listen(PORT, () => {
  console.log(`server listening on port ${PORT}`);
});
