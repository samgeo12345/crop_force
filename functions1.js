function contact_submit(){
    alert(`Your submission has been received !`)
    window.location.href="index.html#Contact_us";
    document.getElementById("contact_form").reset();
}
/*
document.addEventListener('scroll', function() {
    const elements = document.querySelectorAll('.Team_member_title, .Team_members');
        
    elements.forEach(element => {
         const elementPosition = element.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;

            // Check if the element is in the viewport
        if (elementPosition < windowHeight && elementPosition > 0) {
            element.style.animation = 'fadeIn 3s ease-in forwards';
        } else {
            element.style.animation = 'fadeOut 3s ease-out forwards';
        }
    });
});
*/


window.onload = function() {
    setTimeout(function() {
      document.getElementById('loader').style.display = 'none';  // Hide loader
      document.getElementById('content').style.display = 'block';  // Show content
    }, 1200);  // seconds
  };


  function logout() {
    if (confirm("Are you sure you want to log out now?")) {
        window.location.href = "login.html";
    } 
}

document.addEventListener("DOMContentLoaded", function() {
const upArrow = document.getElementById("upArrow");

// Add a scroll event listener
window.addEventListener("scroll", function() {
if (window.scrollY > 300) { // When scrolls more than 300px from the top icon shows
    upArrow.style.display = "block"; 
} else {
    upArrow.style.display = "none"; 
}
});

upArrow.addEventListener("click", function() {
window.scrollTo({ top: 0, behavior: 'smooth' });
});

});


function Sub_Alert(){
    alert('You have been subscribed!');
    window.location.href="index.html#Sub_form";
    document.getElementById("Sub_form").reset();
}

// Date and time
function updateTime() {
    const now = new Date();

    const day = String(now.getDate()).padStart(2, '0');
    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const month = monthNames[now.getMonth()];
    const year = now.getFullYear();

    let hours = now.getHours();
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';

    hours = hours % 12 || 12; 
    const timeString = `${String(hours).padStart(2, '0')}:${minutes} ${ampm}`;

    const dateString = `${day}-${month}-${year}`;
    const dateTimeString = `Current time: ${dateString} ${timeString} IST`;

    document.getElementById('clock').textContent = dateTimeString;
}
setInterval(updateTime, 1000);
updateTime();