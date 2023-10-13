var options = {
  series: [confidence_level],
  chart: {
      height: 350,
      type: 'radialBar',
      offsetY: -40,
  },
  plotOptions: {
      radialBar: {
          startAngle: -135,
          endAngle: 135,
          dataLabels: {
              name: {
                  fontSize: '22px',
                  color: '#148506',
                  offsetY: 120
              },
              value: {
                  offsetY: -10,
                  fontSize: '30px',
                  color: '#148506',
                  formatter: function (val) {
                      return val + "%";
                  }
              }
          }
      }
  },
  fill: {
      type: 'solid', // Change type to 'solid'
      colors: '#148506', // Change the color here
  },
  stroke: {
      dashArray: 5
  },
  labels: ['Confidence Level'],
};

var chart = new ApexCharts(document.querySelector("#chartDiv"), options);
chart.render();
