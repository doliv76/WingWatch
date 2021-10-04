$(document).ready(function(){
    $(".upload-button").click(function(){
        $("#fileId").click();
    })
    $('#predict').prop("disabled",true);
    $("#fileId").change(function(){
        $("#fileId").click();
        var value = $("#fileId").val();
        if (value != ""){
            $("#labelId").html(value.split('\\').pop());
        } else {
            $("#labelId").html("Choose an image file");
        }
        $('#predict').prop("disabled",false);
    })

    $('#predict').click(function() {
        event.preventDefault();
        $('#result p').text('Please wait...')
        var form_data = new FormData($('#formId')[0]);
        $.ajax({
            type: 'POST',
            url: '/upload',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log(data);
                $("#labelId").html("Choose an image file");
                $('#result p').text(data);
                $('#predict').prop("disabled",true);
            }
        })
    });
    
});