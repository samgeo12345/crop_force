$(document).ready(function() {
    $(window).on('scroll', function() {
        var teamSection = $('.Team_members,.Team_member_title');
        var scrollPosition = $(window).scrollTop();
        var windowHeight = $(window).height();
        var teamOffset = teamSection.offset().top;
        var teamHeight = teamSection.outerHeight();
        
        // Section is fully in view
        if (scrollPosition + windowHeight >= teamOffset + (teamHeight / 1)) {
            teamSection.addClass('fade-in').removeClass('fade-out');
        } 
        // Section is scrolled out of view (from top)
        else if (scrollPosition + windowHeight < teamOffset || scrollPosition > teamOffset + teamHeight) {
            teamSection.addClass('fade-out').removeClass('fade-in');
        }
    });
});



