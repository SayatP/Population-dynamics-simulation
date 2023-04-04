function createGrid(numRows, numCols, dataPoints) {
    const canvas = document.querySelector('canvas');
    const ctx = canvas.getContext('2d');

    // Set the canvas size to match the window size
    canvas.width = window.innerWidth-20;
    canvas.height = window.innerHeight-20;

    // Calculate the cell size based on the number of rows/columns
    const cellWidth = canvas.width / numCols;
    const cellHeight = canvas.height / numRows;

    // Draw the background
    ctx.fillStyle = '#E5FFCC';
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
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.stroke();

    // Draw the data points
    for (let i = 0; i < dataPoints.length; i++) {
        const [row, col, type] = dataPoints[i];
        const x = col * cellWidth;
        const y = row * cellHeight;
        if (type === 0) {
        ctx.fillStyle = 'black';
        } else if (type === 1) {
        ctx.fillStyle = 'blue';
        } else if (type === 2) {
        ctx.fillStyle = 'red';
        }
        ctx.fillRect(x, y, cellWidth, cellHeight);
    }

}
  
  
data_points1 = [[0,0,1], [10,20,0], [49,49,2], [20,20,1]];
data_points2 = [[0,1,1], [10,21,0], [48,49,2], [20,19,1]];
data_points3 = [[1,1,1], [11,21,0], [47,49,2], [20,18,1]];

  
createGrid(50, 75, data_points1);

setTimeout(function() {
    createGrid(50, 75, data_points2);
}, 1000);

setTimeout(function() {
    createGrid(50, 75, data_points3);
}, 2000);