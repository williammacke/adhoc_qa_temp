var key = ""
var allText = ""

function generateKey(){
    const length = 16;
    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";  
    const charactersLength = characters.length;
    for ( var i = 0; i < length; i++ ) {
        key += characters.charAt(Math.floor(Math.random() * charactersLength));
    }
}

function resetTarget() {
    var selector = Sk.TurtleGraphics.target;
    var target =
        typeof selector === "string"
            ? document.getElementById(selector)
            : selector;
    // clear canvas container
    while (target.firstChild) {
        target.removeChild(target.firstChild);
    }
    return target;
}

function builtinRead(x) {
    if (
        Sk.builtinFiles === undefined ||
        Sk.builtinFiles["files"][x] === undefined
    )
        throw "File not found: '" + x + "'";
    return Sk.builtinFiles["files"][x];
}

function printString(text) {
    var filename = key + ".txt";

    console.log(text);

    if(text == "date" || text == "done"){
        text = text == "done" ? allText + "\n" : text
        allText = ""

        $.ajax({
            url : 'cgi-bin/writeFile.cgi',
            type : 'post',
            data : {data : text, filename : filename}
        }); 


    } else if (text == "complete"){
        var buttonCode = document.getElementById("code");
        buttonCode.style.display = "inline"
        var codeText = document.getElementById("codeText");
        codeText.style.display = "inline"
        codeText.innerHTML = key
    } else {
        allText += text;
    }
}

function getCode(){  
    var textarea = document.createElement('textarea');
    textarea.textContent = key;
    document.body.appendChild(textarea);

    var selection = document.getSelection();
    var range = document.createRange();
    //  range.selectNodeContents(textarea);
    range.selectNode(textarea);
    selection.removeAllRanges();
    selection.addRange(range);
    
    document.execCommand('copy')
    selection.removeAllRanges();

    document.body.removeChild(textarea);

    alert("You have successfully copied your MTurk code: \n" + key);
}

function addModal() {
    $("#mycanvas").css("height", window.innerHeight * 0.95);
    $("#mycanvas").css("width", window.innerWidth * 0.65);
    $("#mycanvas").css("margin", "auto");
    $(Sk.main_canvas).css("height", window.innerHeight * 0.95);
    $(Sk.main_canvas).css("width", window.innerWidth * 0.65);
    $(Sk.main_canvas).css("margin", "auto");
    $(Sk.main_canvas).css("border", "1px solid blue");

    // $("#myCanvas").append(Sk.main_canvas);
    var currentTarget = resetTarget();
    currentTarget.append(Sk.main_canvas);

    // var div1 = document.createElement("div");
    // currentTarget.appendChild(div1);
    // $(div1).addClass("modal");
    // $(div1).css("text-align", "center");

    // var btn1 = document.createElement("span");
    // $(btn1).addClass("btn btn-primary btn-sm pull-right");
    // var ic = document.createElement("i");
    // $(ic).addClass("fas fa-times");
    // btn1.appendChild(ic);

    // $(btn1).on("click", function (e) {
    //     Sk.insertEvent("quit");
    // });

    // var div2 = document.createElement("div");
    // $(div2).addClass("modal-dialog modal-lg");
    // $(div2).css("display", "inline-block");
    // $(div2).width(self.width + 42);
    // $(div2).attr("role", "document");
    // div1.appendChild(div2);

    // var div3 = document.createElement("div");
    // $(div3).addClass("modal-content");
    // div2.appendChild(div3);

    // var div4 = document.createElement("div");
    // $(div4).addClass("modal-header d-flex justify-content-between");
    // var div5 = document.createElement("div");
    // $(div5).addClass("modal-body");
    // var div6 = document.createElement("div");
    // $(div6).addClass("modal-footer");
    // var div7 = document.createElement("div");
    // $(div7).addClass("col-md-8");
    // var div8 = document.createElement("div");
    // $(div8).addClass("col-md-4");
    // var header = document.createElement("h5");
    // Sk.title_container = header;
    // $(header).addClass("modal-title");

    // div3.appendChild(div4);
    // div3.appendChild(div5);
    // div3.appendChild(div6);

    // div4.appendChild(header);
    // div4.appendChild(btn1);
    // // div7.appendChild(header);
    // // div8.appendChild(btn1);

    // div5.appendChild(Sk.main_canvas);

    // createArrows(div6);
    // $(div1).modal({
    //     backdrop: "static",
    //     keyboard: false,
    // });
}
