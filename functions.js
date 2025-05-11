function show_password(){
    var x=document.getElementById("password")
    if (x.type=="password"){
        x.type="text"
    }
    else{
        x.type="password"
    }
}
function show_pass(){
    var x=document.getElementById("Password")
    var y=document.getElementById("confirm_passsword")
    if (x.type=="password" && y.type=="password"){
        x.type="text"
        y.type="text"
    }
    else{
        x.type="password"
        y.type="password"
    }
}
function match_password(){
    var a=document.getElementById("Password").value;
    var b=document.getElementById("confirm_passsword").value;
    var c=document.getElementById("pass_message")
    var d=document.getElementById("button")
    var e=document.getElementById("reset_button")
    if (a==b&&b==a){
        c.textContent = '';
        c.hidden;
        d.disabled = false;
        d.style.cursor = "pointer";                    
    }
    else{
        c.textContent = 'Password not matched !';
        c.style.color = 'red';
        d.disabled = true;   
        d.style.cursor = "not-allowed";
    }
};
function match_pass(){
    var a=document.getElementById("Password").value;
    var b=document.getElementById("confirm_passsword").value;
    var c=document.getElementById("pass_message")
    var e=document.getElementById("reset_button")
    if (a==b){
        c.textContent = '';
        c.hidden;
        e.disabled = false;
        e.style.cursor = "pointer";                     
    }
    else{
        c.textContent = 'Password not matched !';
        c.style.color = 'red';
        e.disabled = true;   
        e.style.cursor = "not-allowed";
         
    }
};
function dashboard_open(event) {
    event.preventDefault(); 
    setTimeout(function() {
        window.location.href = "dashboard.html"; 
    }, 500); // Redirects after 1 second
}
function submit_Alert(){
    alert(`Successfully Registered! Please Login`);
}
function Login_open(event) {
    event.preventDefault(); 
    setTimeout(function() {
        window.location.href = "login.html"; 
    }, 500); // Redirects after 1 second
}
function reset_alert(){
    alert(`Your password has been changed! Please Login`);
    window.location.href="login.html";
}
var otp_disp;
function OTP_send(){
    var n=document.getElementById("Phone_number").value;
    if (n==""){
        var o=document.getElementById("disp_message");
        o.innerHTML="Enter valid phone number";
        o.style.color= "red"
    }
    else if (/^[0-5]/.test(n)){
        var o=document.getElementById("disp_message");
        o.innerHTML="Enter valid phone number";
        o.style.color= "red"
    }
    else{
        var o=document.getElementById("disp_message").innerHTML="OTP send Successfully !";
        var e=document.getElementById("Verify2_btn");
        var otp=document.getElementById("OTP_in");
        e.disabled=false;
        e.style.cursor="pointer"
        otp.disabled=false;
        otp_disp= Math.floor(100000 + Math.random() * 900000);
        otp_message=document.getElementById("Otp_disp").innerHTML="Your OTP to reset password for +91 "+n+ " is "+otp_disp;
        otp_message.style.visibility="visible";
    }
}
function validate_OTP(){
    var OTP_enter=document.getElementById("OTP_in").value;
    var d=document.getElementById("reset_button");
    var o=document.getElementById("disp_message");
    var password=document.getElementById("Password");
    var confirm_passsword=document.getElementById("confirm_passsword");
    var otp_message=document.getElementById("Otp_disp")
    if (otp_disp==OTP_enter){
        o.innerHTML="OTP verified !"
        o.style.color="green";
        d.disabled=false;
        password.disabled=false;
        confirm_passsword.disabled=false;
        otp_message.style.visibility="hidden";
    }
    else {
        var o=document.getElementById("disp_message");
        o.innerHTML="Invalid OTP !"
        o.style.color="red"
        password.disabled=true;
        confirm_passsword.disabled=true;
    }
}
document.addEventListener("DOMContentLoaded",function(){
    const tools_toggle=document.getElementById("tools_toggle");
    const tools=document.getElementById("tools");
    tools_toggle.onclick=function(){
        if(tools.style.opacity=="1"){
            tools.style.opacity="0";
            tools.style.right="-100%";
        }
        else{
            tools.style.opacity="1";
            tools.style.right="15%";
        }
    }
});

document.addEventListener("DOMContentLoaded",function(){
    const barss=document.getElementById("barss");
    const closebarss=document.getElementById("closebarss");
    const sidebarss=document.getElementById("sidebarss");
    barss.onclick=function(){
        sidebarss.style.display="block";
        sidebarss.style.position="fixed";
    }
    closebarss.onclick=function(){
        sidebarss.style.display="none";
    }
});