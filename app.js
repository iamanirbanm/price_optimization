const submitForm = () => {
    var formData = new FormData(document.getElementById('pricing-form'));
    alert("Data Sent to the Server for Processing...")
    fetch('http://127.0.0.1:3000/engine', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
