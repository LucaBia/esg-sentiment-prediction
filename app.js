$(document).ready(function() {
    $('#stockForm').on('submit', function(e) {
        e.preventDefault();
        const ticker = $('#ticker').val().toUpperCase();
        
        if (ticker) {
            $.ajax({
                url: 'https://esg-sentiment-prediction.onrender.com/predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ ticker: ticker }),
                success: function(response) {
                    const currentDate = new Date();
                    const nextMonth = currentDate.toLocaleString('default', { month: 'long' });

                    $('#results').html(`
                        <div class="result-box">
                            <p>Predicted Monthly Return for ${nextMonth}:</p>
                            <h4>${(response.predicted_monthly_return*100).toFixed(4)}%</h4>
                        </div>
                        <div class="result-box">
                            <p>Actual Monthly Return:</p>
                            <h4>${(response.actual_monthly_return*100).toFixed(4)}%</h4>
                        </div>
                    `);
                },
                error: function(error) {
                    $('#results').html(`
                        <div class="alert alert-danger">
                            Error fetching data. Please try again.
                        </div>
                    `);
                }
            });
        } else {
            $('#results').html(`
                <div class="alert alert-warning">
                    Please enter a valid ticker symbol.
                </div>
            `);
        }
    });
});