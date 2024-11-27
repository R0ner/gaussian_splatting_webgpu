// OBJParser.js from OBJViewer.js (c) 2012 matsuda and itami
//
// Modified by Jeppe Revall Frisvad, 2014, in order to
// - enable loading of OBJ files with no object or group names,
// - enable loading of files with different white spaces and returns at the end
//   of the face definitions, and
// - enable loading of larger models by improving the function getDrawingInfo.
// Modified by Jeppe Revall Frisvad 2023, in order to
// - pad to vec4 for saving in storage buffers,
// - extend loading of materials to include a material index array,
// - extend loading of material properties to more than just Kd, and
// - extract indices of triangles with emission.
// Modified by Jeppe Revall Frisvad 2024, in order to
// - use fetch for asynchronous data loading.

//------------------------------------------------------------------------------
// OBJParser
//------------------------------------------------------------------------------

async function readOBJFile(fileName, scale, reverse)
{
  const response = await fetch(fileName);
  if(response.ok)
  {
    var objDoc = new OBJDoc(fileName); // Create an OBJDoc object
    let fileText = await response.text();
    let result = await objDoc.parse(fileText, scale, reverse);
    if(!result) {
      console.log("OBJ file parsing error.");
      return null;
    }
    return objDoc.getDrawingInfo();
  }
  else
    return null;
}

// OBJDoc object
// Constructor
var OBJDoc = function (fileName) {
  this.fileName = fileName;
  this.mtls = new Array(0);      // Initialize the property for MTL
  this.objects = new Array(0);   // Initialize the property for Object
  this.vertices = new Array(0);  // Initialize the property for Vertex
  this.normals = new Array(0);   // Initialize the property for Normal
}

// Parsing the OBJ file
OBJDoc.prototype.parse = async function (fileString, scale, reverse) {
  var lines = fileString.split('\n');  // Break up into lines and store them as array
  lines.push(null); // Append null
  var index = 0;    // Initialize index of line

  var currentObject = new OBJObject("");
  this.objects.push(currentObject);
  var currentMaterialName = "";

  // Parse line by line
  var line;         // A string in the line to be parsed
  var sp = new StringParser();  // Create StringParser
  while ((line = lines[index++]) != null) {
    sp.init(line);                  // init StringParser
    var command = sp.getWord();     // Get command
    if (command == null) continue;  // check null command

    switch (command) {
      case '#':
        continue;  // Skip comments
      case 'mtllib':     // Read Material chunk
        var path = this.parseMtllib(sp, this.fileName);
        var mtl = new MTLDoc();   // Create MTL instance
        this.mtls.push(mtl);
        const response = await fetch(path);
        if(response.ok) {
          onReadMTLFile(await response.text(), mtl);
        }
        else {
          mtl.complete = true;
        }
        continue; // Go to the next line
      case 'o':
      case 'g':   // Read Object name
        if (currentObject.numIndices == 0) {
          currentObject = this.parseObjectName(sp);
          this.objects[0] = currentObject;
        }
        else {
          var object = this.parseObjectName(sp);
          this.objects.push(object);
          currentObject = object;
        }
        continue; // Go to the next line
      case 'v':   // Read vertex
        var vertex = this.parseVertex(sp, scale);
        this.vertices.push(vertex);
        continue; // Go to the next line
      case 'vn':   // Read normal
        var normal = this.parseNormal(sp);
        this.normals.push(normal);
        continue; // Go to the next line
      case 'usemtl': // Read Material name
        currentMaterialName = this.parseUsemtl(sp);
        continue; // Go to the next line
      case 'f': // Read face
        var face = this.parseFace(sp, currentMaterialName, this.vertices, reverse);
        currentObject.addFace(face);
        continue; // Go to the next line
    }
  }

  return true;
}

OBJDoc.prototype.parseMtllib = function (sp, fileName) {
  // Get directory path
  var i = fileName.lastIndexOf("/");
  var dirPath = "";
  if (i > 0) dirPath = fileName.substr(0, i + 1);

  return dirPath + sp.getWord();   // Get path
}

OBJDoc.prototype.parseObjectName = function (sp) {
  var name = sp.getWord();
  return (new OBJObject(name));
}

OBJDoc.prototype.parseVertex = function (sp, scale) {
  var x = sp.getFloat() * scale;
  var y = sp.getFloat() * scale;
  var z = sp.getFloat() * scale;
  return (new Vertex(x, y, z));
}

OBJDoc.prototype.parseNormal = function (sp) {
  var x = sp.getFloat();
  var y = sp.getFloat();
  var z = sp.getFloat();
  return (new Normal(x, y, z));
}

OBJDoc.prototype.parseUsemtl = function (sp) {
  return sp.getWord();
}

OBJDoc.prototype.parseFace = function (sp, materialName, vertices, reverse) {
  var face = new Face(materialName);
  // get indices
  for (; ;) {
    var word = sp.getWord();
    if (word == null) break;
    var subWords = word.split('/');
    if (subWords.length >= 1) {
      var vi = parseInt(subWords[0]) - 1;
      if (!isNaN(vi))
        face.vIndices.push(vi);
    }
    if (subWords.length >= 3) {
      var ni = parseInt(subWords[2]) - 1;
      face.nIndices.push(ni);
    } else {
      face.nIndices.push(-1);
    }
  }

  // calc normal
  var v0 = [
    vertices[face.vIndices[0]].x,
    vertices[face.vIndices[0]].y,
    vertices[face.vIndices[0]].z];
  var v1 = [
    vertices[face.vIndices[1]].x,
    vertices[face.vIndices[1]].y,
    vertices[face.vIndices[1]].z];
  var v2 = [
    vertices[face.vIndices[2]].x,
    vertices[face.vIndices[2]].y,
    vertices[face.vIndices[2]].z];

  // 面の法線を計算してnormalに設定
  var normal = calcNormal(v0, v1, v2);
  // 法線が正しく求められたか調べる
  if (normal == null) {
    if (face.vIndices.length >= 4) { // 面が四角形なら別の3点の組み合わせで法線計算
      var v3 = [
        vertices[face.vIndices[3]].x,
        vertices[face.vIndices[3]].y,
        vertices[face.vIndices[3]].z];
      normal = calcNormal(v1, v2, v3);
    }
    if (normal == null) {         // 法線が求められなかったのでY軸方向の法線とする
      normal = [0.0, 1.0, 0.0];
    }
  }
  if (reverse) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
  face.normal = new Normal(normal[0], normal[1], normal[2]);

  // Devide to triangles if face contains over 3 points.
  if (face.vIndices.length > 3) {
    var n = face.vIndices.length - 2;
    var newVIndices = new Array(n * 3);
    var newNIndices = new Array(n * 3);
    for (var i = 0; i < n; i++) {
      newVIndices[i * 3 + 0] = face.vIndices[0];
      newVIndices[i * 3 + 1] = face.vIndices[i + 1];
      newVIndices[i * 3 + 2] = face.vIndices[i + 2];
      newNIndices[i * 3 + 0] = face.nIndices[0];
      newNIndices[i * 3 + 1] = face.nIndices[i + 1];
      newNIndices[i * 3 + 2] = face.nIndices[i + 2];
    }
    face.vIndices = newVIndices;
    face.nIndices = newNIndices;
  }
  face.numIndices = face.vIndices.length;

  return face;
}

// Analyze the material file
function onReadMTLFile(fileString, mtl) {
  var lines = fileString.split('\n');  // Break up into lines and store them as array
  lines.push(null);           // Append null
  var index = 0;              // Initialize index of line

  // Parse line by line
  var material;
  var line;      // A string in the line to be parsed
  var name = ""; // Material name
  var sp = new StringParser();  // Create StringParser
  while ((line = lines[index++]) != null) {
    sp.init(line);                  // init StringParser
    var command = sp.getWord();     // Get command
    if (command == null) continue;  // check null command

    switch (command) {
      case '#':
        continue;    // Skip comments
      case 'newmtl': // Read Material chunk
        name = mtl.parseNewmtl(sp);    // Get name
        material = new Material(name, 0.8, 0.8, 0.8, 1.0);
        mtl.materials.push(material);
        continue;  // Go to the next line
      case 'Kd':   // Read diffuse color coefficient as color
        //if (name == "") continue; // Go to the next line because of Error
        if(material)
          material.color = mtl.parseRGB(sp);
        name = "";
        continue;  // Go to the next line
      case 'Ka':   // Read ambient color coefficient as emission
        if(material)
          material.emission = mtl.parseRGB(sp);
        name = "";
        continue;  // Go to the next line
      case 'Ks':   // Read specular color coefficient
        if(material)
          material.specular = mtl.parseRGB(sp);
        name = "";
        continue;  // Go to the next line
      case 'Ni':   // Read specular color coefficient
        if(material)
          material.ior = sp.getFloat();
        name = "";
        continue;  // Go to the next line
      case 'Ns':   // Read specular color coefficient
        if(material)
          material.shininess = sp.getFloat();
        name = "";
        continue;  // Go to the next line
      case 'illum':   // Read specular color coefficient
        if(material)
          material.illum = sp.getInt();
        name = "";
        continue;  // Go to the next line
    }
  }
  mtl.complete = true;
}

// Check Materials
OBJDoc.prototype.isMTLComplete = function () {
  if (this.mtls.length == 0) return true;
  for (var i = 0; i < this.mtls.length; i++) {
    if (!this.mtls[i].complete) return false;
  }
  return true;
}

// Find color by material name
OBJDoc.prototype.findMaterial = function (name) {
  for (var i = 0; i < this.mtls.length; i++) {
    for (var j = 0; j < this.mtls[i].materials.length; j++) {
      if (this.mtls[i].materials[j].name == name) {
        return this.mtls[i].materials[j];
      }
    }
  }
  return (new Color(0.8, 0.8, 0.8, 1));
}

//------------------------------------------------------------------------------
// Retrieve the information for drawing 3D model
OBJDoc.prototype.getDrawingInfo = function () {
  // Create an arrays for vertex coordinates, normals, colors, and indices
  var numVertices = 0;
  var numIndices = 0;
  var numFaces = 0;
  for (var i = 0; i < this.objects.length; i++) {
    numIndices += this.objects[i].numIndices + this.objects[i].faces.length;
    numFaces += this.objects[i].faces.length;
  }
  var numVertices = this.vertices.length;
  var vertices = new Float32Array(numVertices * 4);
  var normals = new Float32Array(numVertices * 4);
  var colors = new Float32Array(numVertices * 4);
  var indices = new Uint32Array(numIndices);
  var mat_indices = new Uint32Array(numFaces);
  var materials = [];
  var mat_map = new Map();
  var light_indices = [];

  // Set vertex, normal and color
  var index_indices = 0;
  var face_indices = 0;
  for (var i = 0; i < this.objects.length; i++) {
    var object = this.objects[i];
    for (var j = 0; j < object.faces.length; j++) {
      var face = object.faces[j];
      var mat_idx = mat_map.get(face.materialName);
      var mat;
      if(mat_idx === undefined) {
        mat = this.findMaterial(face.materialName);
        mat_map.set(face.materialName, materials.length);
        mat_idx = materials.length;
        materials.push(mat);
      }
      else {
        mat = materials[mat_idx];
      }
      if(mat.emission !== undefined && mat.emission.r + mat.emission.g + mat.emission.b > 0.0)
        light_indices.push(face_indices);
      mat_indices[face_indices++] = mat_idx;
      var color = mat.color === undefined ? new Color(0.8, 0.8, 0.8, 1.0) : mat.color;
      var faceNormal = face.normal;
      for (var k = 0; k < face.vIndices.length; k++) {
        // Set index
        var vIdx = face.vIndices[k];
        indices[index_indices] = vIdx;
        // Copy vertex
        var vertex = this.vertices[vIdx];
        vertices[vIdx * 4 + 0] = vertex.x;
        vertices[vIdx * 4 + 1] = vertex.y;
        vertices[vIdx * 4 + 2] = vertex.z;
        vertices[vIdx * 4 + 3] = 1.0;
        // Copy color
        colors[vIdx * 4 + 0] = color.r;
        colors[vIdx * 4 + 1] = color.g;
        colors[vIdx * 4 + 2] = color.b;
        colors[vIdx * 4 + 3] = color.a;
        // Copy normal
        var nIdx = face.nIndices[k];
        if (nIdx >= 0) {
          var normal = this.normals[nIdx];
          normals[vIdx * 4 + 0] = normal.x;
          normals[vIdx * 4 + 1] = normal.y;
          normals[vIdx * 4 + 2] = normal.z;
          normals[vIdx * 4 + 3] = 0.0;
        } else {
          normals[vIdx * 4 + 0] = faceNormal.x;
          normals[vIdx * 4 + 1] = faceNormal.y;
          normals[vIdx * 4 + 2] = faceNormal.z;
          normals[vIdx * 4 + 3] = 0.0;
        }
        index_indices++;
      }
      indices[index_indices++] = mat_idx;
    }
  }

  return new DrawingInfo(vertices, normals, colors, indices, materials, mat_indices, new Uint32Array(light_indices));
}

//------------------------------------------------------------------------------
// MTLDoc Object
//------------------------------------------------------------------------------
var MTLDoc = function () {
  this.complete = false; // MTL is configured correctly
  this.materials = new Array(0);
}

MTLDoc.prototype.parseNewmtl = function (sp) {
  return sp.getWord();         // Get name
}

MTLDoc.prototype.parseRGB = function (sp, name) {
  var r = sp.getFloat();
  var g = sp.getFloat();
  var b = sp.getFloat();
  return new Color(r, g, b, 1);
}

//------------------------------------------------------------------------------
// Material Object
//------------------------------------------------------------------------------
var Material = function (name, r, g, b, a) {
  this.name = name;
  this.color = new Color(r, g, b, a);
}

//------------------------------------------------------------------------------
// Vertex Object
//------------------------------------------------------------------------------
var Vertex = function (x, y, z) {
  this.x = x;
  this.y = y;
  this.z = z;
}

//------------------------------------------------------------------------------
// Normal Object
//------------------------------------------------------------------------------
var Normal = function (x, y, z) {
  this.x = x;
  this.y = y;
  this.z = z;
}

//------------------------------------------------------------------------------
// Color Object
//------------------------------------------------------------------------------
var Color = function (r, g, b, a) {
  this.r = r;
  this.g = g;
  this.b = b;
  this.a = a;
}

//------------------------------------------------------------------------------
// OBJObject Object
//------------------------------------------------------------------------------
var OBJObject = function (name) {
  this.name = name;
  this.faces = new Array(0);
  this.numIndices = 0;
}

OBJObject.prototype.addFace = function (face) {
  this.faces.push(face);
  this.numIndices += face.numIndices;
}

//------------------------------------------------------------------------------
// Face Object
//------------------------------------------------------------------------------
var Face = function (materialName) {
  this.materialName = materialName;
  if (materialName == null) this.materialName = "";
  this.vIndices = new Array(0);
  this.nIndices = new Array(0);
}

//------------------------------------------------------------------------------
// DrawInfo Object
//------------------------------------------------------------------------------
var DrawingInfo = function (vertices, normals, colors, indices, materials, mat_indices, light_indices) {
  this.vertices = vertices;
  this.normals = normals;
  this.colors = colors;
  this.indices = indices;
  this.materials = materials;
  this.mat_indices = mat_indices;
  this.light_indices = light_indices;
}

//------------------------------------------------------------------------------
// Constructor
var StringParser = function (str) {
  this.str;   // Store the string specified by the argument
  this.index; // Position in the string to be processed
  this.init(str);
}
// Initialize StringParser object
StringParser.prototype.init = function (str) {
  this.str = str;
  this.index = 0;
}

// Skip delimiters
StringParser.prototype.skipDelimiters = function () {
  for (var i = this.index, len = this.str.length; i < len; i++) {
    var c = this.str.charAt(i);
    // Skip TAB, Space, '(', ')
    if (c == '\t' || c == ' ' || c == '(' || c == ')' || c == '"') continue;
    break;
  }
  this.index = i;
}

// Skip to the next word
StringParser.prototype.skipToNextWord = function () {
  this.skipDelimiters();
  var n = getWordLength(this.str, this.index);
  this.index += (n + 1);
}

// Get word
StringParser.prototype.getWord = function () {
  this.skipDelimiters();
  var n = getWordLength(this.str, this.index);
  if (n == 0) return null;
  var word = this.str.substr(this.index, n);
  this.index += (n + 1);

  return word;
}

// Get integer
StringParser.prototype.getInt = function () {
  return parseInt(this.getWord());
}

// Get floating number
StringParser.prototype.getFloat = function () {
  return parseFloat(this.getWord());
}

// Get the length of word
function getWordLength(str, start) {
  var n = 0;
  for (var i = start, len = str.length; i < len; i++) {
    var c = str.charAt(i);
    if (c == '\t' || c == ' ' || c == '(' || c == ')' || c == '"')
      break;
  }
  return i - start;
}

//------------------------------------------------------------------------------
// Common function
//------------------------------------------------------------------------------
function calcNormal(p0, p1, p2) {
  // v0: a vector from p1 to p0, v1; a vector from p1 to p2
  var v0 = new Float32Array(3);
  var v1 = new Float32Array(3);
  for (var i = 0; i < 3; i++) {
    v0[i] = p0[i] - p1[i];
    v1[i] = p2[i] - p1[i];
  }

  // The cross product of v0 and v1
  var c = new Float32Array(3);
  c[0] = v0[1] * v1[2] - v0[2] * v1[1];
  c[1] = v0[2] * v1[0] - v0[0] * v1[2];
  c[2] = v0[0] * v1[1] - v0[1] * v1[0];

  var x = c[0], y = c[1], z = c[2], g = Math.sqrt(x * x + y * y + z * z);
  if (g) {
    if (g == 1)
      return c;
  } else {
    c[0] = 0; c[1] = 0; c[2] = 0;
    return c;
  }
  g = 1 / g;
  c[0] = x * g; c[1] = y * g; c[2] = z * g;
  return c;
}
