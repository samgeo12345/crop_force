const apiKey = '3652485fbeff1136836d4e26e5d0557c'; // Replace with your actual API key
        const weatherDiv = document.getElementById('weather');

        // Get user's current location
        function getLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(showWeather);
            } else {
                weatherDiv.innerHTML = "Geolocation is not supported by this browser.";
            }
        }

        // Fetch weather data using OpenWeatherMap API
        function showWeather(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;

            fetch(url)
                .then(response => response.json())
                .then(data => {
                    // Extract location data
                    const city = data.name;
                    const country = data.sys.country;
                    const weatherDescription = data.weather[0].description; // Get the weather description

                    // Extract other weather details
                    const temperature = `${data.main.temp} °C`;
                    const pressure = `${data.main.pressure} hPa`;
                    const humidity = `${data.main.humidity} %`;
                    const fog = data.weather.some(weather => weather.description.includes('fog')) ? 'Yes' : 'No';
                    const rain = data.hasOwnProperty('rain') ? 'Yes' : 'No';
                    const sunny = data.weather.some(weather => weather.description.includes('clear')) ? 'Yes' : 'No';

                    // Combine data for display
                    const location = `${city}, ${country}`;
                    document.getElementById('location').innerText = `Location: ${location}`;
                    document.getElementById('temperature').innerText = `Temperature: ${temperature}`;
                    document.getElementById('pressure').innerText = `Pressure: ${pressure}`;
                    document.getElementById('humidity').innerText = `Humidity: ${humidity}`;
                    document.getElementById('fog').innerText = `Fog: ${fog}`;
                    document.getElementById('rain').innerText = `Rain Possibility: ${rain}`;
                    document.getElementById('sunny').innerText = `Sunny: ${sunny}`;
                    document.getElementById('weatherDescription').innerText = `Weather: ${weatherDescription.charAt(0).toUpperCase() + weatherDescription.slice(1)}`;
                })
                .catch(error => {
                    weatherDiv.innerHTML = "Error fetching weather data.";
                    console.error("Error:", error);
                });
        }

        // Initialize the app
        getLocation();