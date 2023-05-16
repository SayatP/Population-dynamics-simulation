function createGrid(data) {
  const canvas = document.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  // Set the canvas size to match the window size
  canvas.width = window.innerWidth - 20;
  canvas.height = window.innerHeight - 20;

  // Calculate the cell size based on the number of rows/columns
  const numRows = data.length;
  const numCols = data[0].length;
  const cellWidth = canvas.width / numCols;
  const cellHeight = canvas.height / numRows;

  // Draw the background
  ctx.fillStyle = "lightgrey";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw the grid
  ctx.beginPath();
  for (let i = 0; i <= numCols; i++) {
    const x = i * cellWidth;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
  }
  for (let i = 0; i <= numRows; i++) {
    const y = i * cellHeight;
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
  }

  // Set the line style properties
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.2)";
  ctx.stroke();

  // Draw the data points
  for (let i = 0; i < numRows; i++) {
    for (let j = 0; j < numCols; j++) {
      const type = data[i][j];
      if (type === 1) {
        ctx.fillStyle = "blue";
      } else if (type === 2) {
        ctx.fillStyle = "red";
      } else {
        continue; // skip cells with type 0 (unoccupied)
      }
      const x = j * cellWidth;
      const y = i * cellHeight;
      ctx.fillRect(x, y, cellWidth, cellHeight);
    }
  }
}

async function main() {
  data = await fetch("./history.json").then((x) => x.json());
  for (let i = 0; i < data.length; i++) {
    setTimeout(function () {
      createGrid(data[i]);
    }, 1000 * i);
  }
}

main();
