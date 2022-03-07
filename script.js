let ultData
let fprRow = [1,2,3,4,5]; 
let mprRow; 
let currData
let qwerCountryKey
let gniChart
let bubbleChart
let bullet1
let bullet2
let bullet3
let bullet4
let pieGraph
let seasonGraphh

window.onload = function () {
    function featureListener () {
      let inputData = document.getElementById('selectDataset').value
      console.log("we are changing now to " + inputData)

      changeContent(inputData)
    }
    let eFeature = document.getElementById('selectDataset')
  eFeature.addEventListener('change', featureListener)



    d3.csv('./fairness_df.csv').then(data => {
      ultData = data
      console.log(ultData)
      Highcharts.setOptions({
        lang: {    
          thousandsSep: ','
        },
        colors: [
          '#058DC7',
          '#50B432',
          '#ED561B',
          '#DDDF00',
          '#24CBE5',
          '#64E572',
          '#FF9655',
          '#FFF263',
          '#6AF9C4'
        ]
      })
      // STEP 1: start off with everything 'ALL'

      currentSelectedDataset = document.getElementById('selectDataset').value;
      console.log('hmhm')
      console.log(currentSelectedDataset)
      changeContent(currentSelectedDataset)


    })
  }

  

function changeContent(datasetName) {
    console.log("Chnaging content to dataset name: " + datasetName);

    //must add column names 'dataset' and 'model'
    if(datasetName == 'all') {
        //continue
        currData = ultData
        xAxisRow = d3.map(ultData, function(d){return( d.dataset + ' ' + d.model)});


    }
    else{
        currData = ultData
        console.log(currData)
        currData = currData.filter(function(d){ return d.dataset == datasetName })
        console.log(currData)
        xAxisRow = d3.map(currData, function(d){return( d.model)});

    }

 
   // chk = d3.rollup(currData, v => d3.sum(v, d => d['Country Name:']), d => d['Indicator Name:'])
  // console.log(chk)
  console.log("Getting infos")
//   let aha = d3.groups(currData, d => d['FPR'])

fprRow = (d3.map(currData, function(d){return(d.FPR)})).map(Number)
console.log(fprRow)
mprRow = (d3.map(currData, function(d){return(d.max_parity_ratio)})).map(Number)
console.log(mprRow)
eoRow = (d3.map(currData, function(d){return(d.equalized_odds_diff)})).map(Number)
srRow = (d3.map(currData, function(d){return(d.selection_rate )})).map(Number)
accuracyRow = (d3.map(currData, function(d){return(d.accuracy_range  )})).map(Number)
accuracy2Row = (d3.map(currData, function(d){return(d.overall_accuracy  )})).map(Number)
recall_range = (d3.map(currData, function(d){return(d.recall_range  )})).map(Number)
brier_score_range = (d3.map(currData, function(d){return(d.brier_score_range  )})).map(Number)
f1 = (d3.map(currData, function(d){return(d.F1  )})).map(Number)


// model	accuracy_range	max_parity_ratio	equalized_odds_diff	overall_accuracy
	// FPR	F1	recall_range	brier_score_range





//   var allGroup = .keys()

  // let testing =  d3.rollup(aha, v => d3.mean(v, d => d['value'])).toFixed(0)
//   console.log(currData)


  for (var i = 0, len = currData.length; i < len; i++) {


    // currentRow = ultData[i]; 

    // console.log(currentRow);
    // currentDatasetName = currentRow['dataset'];
    // currentModelName = currentRow['model'];

    
    // myArray.push(ultData[i][1])
    // arrayYears.push(ultData[i][0])
    // enrollArray.push(mapEnroll[i][1])
  }




  Highcharts.chart('matrixContainer', {
    chart: {
      type: 'column'
    },
        
  colors: [
    '#182B49', 
    '#C69214', 
    '#00C6D7', 
    '#6E963B', 
    '#F3E500', 
    '#FC8900', 
    '#92A8CD', 
    '#A47D7C', 
    '#B5CA92'
    ],

    plotOptions: {
        column: {
            colorByPoint: true
        }
    },
    title: {
      text: 'Model/Fairness of ' + datasetName
    },
    xAxis: {
      categories: xAxisRow,
      crosshair: true
    },
    yAxis: {
      min: 0,
      title: {
        text: ''
      }
    },
    tooltip: {
      headerFormat: '<span style="font-size:10px">{point.key}</span><table>',
      pointFormat: '<tr><td style="color:{series.color};padding:0">{series.name}: </td>' +
        '<td style="padding:0"><b>{point.y:.3f}</b></td></tr>',
      footerFormat: '</table>',
      shared: true,
      useHTML: true
    },
    plotOptions: {
      column: {
        pointPadding: 0.2,
        borderWidth: 0
      }
    },
    series: [{
      name: 'FPR',
      data: fprRow
    }, {
      name: 'max_parity_ratio',
      data: mprRow
    },  {
        name: 'equalized_odds_diff',
        data: eoRow
      }, {
        name: 'selection rate',
        data: srRow
      }, {
        name: 'accuracy range',
        data: accuracyRow
      }, {
        name: 'overall accuracy',
        data: accuracy2Row
      },
      {
        name: 'recall range',
        data: recall_range
      }, {
        name: 'brier score range',
        data: brier_score_range
      }, {
        name: 'F1 range',
        data: f1
      },



    // , {
    //   name: 'equalized_odds_diff',
    //   data: [48.9, 38.8, 39.3, 41.4, 47.0, 48.3, 59.0, 59.6, 52.4, 65.2, 59.3, 51.2]
  
    // }, {
    //   name: 'selection_rate',
    //   data: [42.4, 33.2, 34.5, 39.7, 52.6, 75.5, 57.4, 60.4, 47.6, 39.1, 46.8, 51.1]
  
    // },  {
    //     name: 'accuracy',
    //     data: [48.9, 38.8, 39.3, 41.4, 47.0, 48.3, 59.0, 59.6, 52.4, 65.2, 59.3, 51.2]
    
    //   }
    
    ]
  });

}



// // Give the points a 3D feel by adding a radial gradient
// Highcharts.setOptions({
//     colors: Highcharts.getOptions().colors.map(function (color) {
//       return {
//         radialGradient: {
//           cx: 0.4,
//           cy: 0.3,
//           r: 0.5
//         },
//         stops: [
//           [0, color],
//           [1, Highcharts.color(color).brighten(-0.2).get('rgb')]
//         ]
//       };
//     })
//   });
  
//   // Set up the chart
//   var chart = new Highcharts.Chart({
//     chart: {
//       renderTo: 'matrixContainer',
//       margin: 100,
//       type: 'scatter3d',
//       animation: false,
//       options3d: {
//         enabled: true,
//         alpha: 10,
//         beta: 30,
//         depth: 250,
//         viewDistance: 5,
//         fitToPlot: false,
//         frame: {
//           bottom: { size: 1, color: 'rgba(0,0,0,0.02)' },
//           back: { size: 1, color: 'rgba(0,0,0,0.04)' },
//           side: { size: 1, color: 'rgba(0,0,0,0.06)' }
//         }
//       }
//     },
//     title: {
//       text: 'Draggable box'
//     },
//     subtitle: {
//       text: 'Click and drag the plot area to rotate in space'
//     },
//     plotOptions: {
//       scatter: {
//         width: 10,
//         height: 10,
//         depth: 10
//       }
//     },
//     yAxis: {
//       min: 0,
//       max: 10,
//       title: null
//     },
//     xAxis: {
//       min: 0,
//       max: 10,
//       gridLineWidth: 1
//     },
//     zAxis: {
//       min: 0,
//       max: 10,
//       showFirstLabel: false
//     },
//     legend: {
//       enabled: false
//     },
//     series: [{
//       name: 'Data',
//       colorByPoint: true,
//       accessibility: {
//         exposeAsGroupOnly: true
//       },
//       data: [
//         [1, 6, 5], [8, 7, 9], [1, 3, 4], [4, 6, 8], [5, 7, 7], [6, 9, 6],
//         [7, 0, 5], [2, 3, 3], [3, 9, 8], [3, 6, 5], [4, 9, 4], [2, 3, 3],
//         [6, 9, 9], [0, 7, 0], [7, 7, 9], [7, 2, 9], [0, 6, 2], [4, 6, 7],
//         [3, 7, 7], [0, 1, 7], [2, 8, 6], [2, 3, 7], [6, 4, 8], [3, 5, 9],
//         [7, 9, 5], [3, 1, 7], [4, 4, 2], [3, 6, 2], [3, 1, 6], [6, 8, 5],
//         [6, 6, 7], [4, 1, 1], [7, 2, 7], [7, 7, 0], [8, 8, 9], [9, 4, 1],
//         [8, 3, 4], [9, 8, 9], [3, 5, 3], [0, 2, 4], [6, 0, 2], [2, 1, 3],
//         [5, 8, 9], [2, 1, 1], [9, 7, 6], [3, 0, 2], [9, 9, 0], [3, 4, 8],
//         [2, 6, 1], [8, 9, 2], [7, 6, 5], [6, 3, 1], [9, 3, 1], [8, 9, 3],
//         [9, 1, 0], [3, 8, 7], [8, 0, 0], [4, 9, 7], [8, 6, 2], [4, 3, 0],
//         [2, 3, 5], [9, 1, 4], [1, 1, 4], [6, 0, 2], [6, 1, 6], [3, 8, 8],
//         [8, 8, 7], [5, 5, 0], [3, 9, 6], [5, 4, 3], [6, 8, 3], [0, 1, 5],
//         [6, 7, 3], [8, 3, 2], [3, 8, 3], [2, 1, 6], [4, 6, 7], [8, 9, 9],
//         [5, 4, 2], [6, 1, 3], [6, 9, 5], [4, 8, 2], [9, 7, 4], [5, 4, 2],
//         [9, 6, 1], [2, 7, 3], [4, 5, 4], [6, 8, 1], [3, 4, 0], [2, 2, 6],
//         [5, 1, 2], [9, 9, 7], [6, 9, 9], [8, 4, 3], [4, 1, 7], [6, 2, 5],
//         [0, 4, 9], [3, 5, 9], [6, 9, 1], [1, 9, 2]]
//     }]
//   });
  
  
