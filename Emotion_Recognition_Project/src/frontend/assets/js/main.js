const API_URL = "http://127.0.0.1:5000/emotion-statistics";

async function fetchEmotionStatistics() {
    try {
        const response = await fetch(API_URL);
        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        renderChart(data["Hào hứng"] || 0, data["Không hào hứng"] || 0);
    } catch (error) {
        console.error("Error fetching data:", error);
        alert("Không thể kết nối với server.");
    }
}

function renderChart(happyPercentage, unhappyPercentage) {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Hào hứng', 'Không hào hứng'],
            datasets: [{
                data: [happyPercentage, unhappyPercentage],
                backgroundColor: ['#36A2EB', '#FF6384'],
                hoverBackgroundColor: ['#36A2EB', '#FF6384']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Tỷ lệ cảm xúc'
                }
            }
        }
    });
}

fetchEmotionStatistics();
