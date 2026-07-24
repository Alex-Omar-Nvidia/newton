function example_brick_CB()
%EXAMPLE_BRICK_CB Export a reduced flexible-brick model in MATLAB or Octave.
%
% This file turns brick.stl into finite-element and reduced-order data that
% Newton can use for a flexible body.  It performs the following operations:
%   1. Read the dimensions of the axis-aligned brick from the STL file.
%   2. Create a structured mesh of first-order tetrahedral elements.
%   3. Assemble the full 3-D linear-elastic stiffness and mass matrices.
%   4. Treat the two end faces as rigid six-DOF attachment interfaces.
%   5. Apply Craig-Bampton component-mode synthesis, retaining the rigid
%      interface coordinates and the two lowest fixed-interface vibration
%      modes.
%   6. Build modal damping and a recovery matrix that maps reduced coordinates
%      back to displacement samples on the brick surface.
%
% The example does not require MATLAB's PDE Toolbox.  It writes three full-FEM
% CSV files and seven Craig-Bampton CSV files beside this script.  The files are
% numeric matrices without headers; see EXPORT_FORMAT.md at the repository root
% for names, shapes, units, and ordering conventions.

format long

scriptDir = fileparts(mfilename('fullpath'));
stlFile = fullfile(scriptDir, 'brick.stl');

% These demonstration values retain the original stiffness while using a
% deliberately higher density to make the inertial deformation easy to see.
E = 69e9;      % Young's modulus [Pa] (original stiffness)
nu = 0.26;     % Poisson's ratio
rho = 10800;   % Mass density [kg/m^3] (2x the previous; 4x original mass)
zeta = 0.05;   % Modal damping ratio
fixedInterfaceModeCount = 2;

% The STL is an axis-aligned rectangular brick.  Its bounds are used instead
% of relying on the unavailable PDE Toolbox geometry and meshing functions.
bounds = readBrickBounds(stlFile);
elementCounts = [5, 2, 2];
[nodes, tets] = createBrickMesh(bounds, elementCounts);
[K, M] = assembleLinearElasticity(nodes, tets, E, nu, rho);

writeCsv(fullfile(scriptDir, 'full_fem_stiffness.csv'), K);
writeCsv(fullfile(scriptDir, 'full_fem_mass.csv'), M);
writeCsv(fullfile(scriptDir, 'full_fem_nodes.csv'), nodes);


%% Craig-Bampton reduction

origins = [bounds(1, 1), mean(bounds(:, 2)), mean(bounds(:, 3));
           bounds(2, 1), mean(bounds(:, 2)), mean(bounds(:, 3))];

[K_ROM, M_ROM, T] = craigBamptonReduction( ...
    K, M, nodes, origins, fixedInterfaceModeCount);
C_ROM = modalDampingMatrix(zeta, K_ROM, M_ROM);
nodes_ROM = origins;


%% Visualization skin and linear displacement recovery map

[F, surfaceNodeIds] = extractSurface(tets, nodes);
P = nodes(surfaceNodeIds, :);
r = size(T, 2);
Nv = size(P, 1);
nodeCount = size(nodes, 1);
Rmat = zeros(3 * Nv, r);

for k = 1:Nv
    node = surfaceNodeIds(k);
    Rmat(3 * k - 2, :) = T(node, :);
    Rmat(3 * k - 1, :) = T(node + nodeCount, :);
    Rmat(3 * k, :) = T(node + 2 * nodeCount, :);
end

writeCsv(fullfile(scriptDir, 'cb_rom_stiffness.csv'), K_ROM);
writeCsv(fullfile(scriptDir, 'cb_rom_mass.csv'), M_ROM);
writeCsv(fullfile(scriptDir, 'cb_rom_damping.csv'), C_ROM);
writeCsv(fullfile(scriptDir, 'cb_rom_interfaces.csv'), nodes_ROM);
writeCsv(fullfile(scriptDir, 'cb_rom_recovery.csv'), Rmat);
writeCsv(fullfile(scriptDir, 'cb_rom_points.csv'), P);
writeCsv(fullfile(scriptDir, 'cb_rom_faces.csv'), F);

fprintf('Wrote full FEM CSV files (%d nodes, %d tetrahedra).\n', ...
    size(nodes, 1), size(tets, 1));
fprintf(['Wrote Craig-Bampton CSV files (%d interface coordinates and ', ...
         '%d fixed-interface modes).\n'], ...
    6 * size(origins, 1), fixedInterfaceModeCount);
end


function bounds = readBrickBounds(filename)
%READBRICKBOUNDS Read and validate the bounding box of a binary or ASCII STL.

info = dir(filename);
if isempty(info)
    error('Could not find STL file: %s', filename);
end

fid = fopen(filename, 'rb');
if fid < 0
    error('Could not open STL file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fseek(fid, 80, 'bof');
triangleCount = fread(fid, 1, 'uint32=>double', 0, 'ieee-le');
isBinary = numel(triangleCount) == 1 && ...
    info.bytes == 84 + 50 * triangleCount;

if isBinary
    vertices = zeros(3 * triangleCount, 3);
    fseek(fid, 84, 'bof');
    for k = 1:triangleCount
        fread(fid, 3, 'float32=>double', 0, 'ieee-le');  % facet normal
        triangle = fread(fid, [3, 3], 'float32=>double', 0, 'ieee-le').';
        if size(triangle, 1) ~= 3
            error('Unexpected end of binary STL file: %s', filename);
        end
        vertices(3 * k - 2:3 * k, :) = triangle;
        fread(fid, 1, 'uint16=>double', 0, 'ieee-le');  % attribute count
    end
else
    frewind(fid);
    contents = fread(fid, Inf, '*char').';
    number = '([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)';
    tokens = regexp(contents, ...
        ['vertex\s+', number, '\s+', number, '\s+', number], 'tokens');
    if isempty(tokens)
        error('The STL file is neither valid binary STL nor valid ASCII STL: %s', filename);
    end
    vertices = zeros(numel(tokens), 3);
    for k = 1:numel(tokens)
        vertices(k, :) = [str2double(tokens{k}{1}), ...
                          str2double(tokens{k}{2}), ...
                          str2double(tokens{k}{3})];
    end
end

bounds = [min(vertices, [], 1); max(vertices, [], 1)];
extent = bounds(2, :) - bounds(1, :);
if any(~isfinite(bounds(:))) || any(extent <= 0)
    error('The STL bounding box is invalid.');
end

% A structured fill of the bounding box is valid only for this brick asset.
tolerance = 1e-6 * max(extent);
for axisIndex = 1:3
    distanceToBound = min(abs(vertices(:, axisIndex) - bounds(1, axisIndex)), ...
                          abs(vertices(:, axisIndex) - bounds(2, axisIndex)));
    if any(distanceToBound > tolerance)
        error(['This example expects an axis-aligned rectangular brick STL. ', ...
               'Use a tetrahedral mesh generator for a general STL geometry.']);
    end
end
end


function [nodes, tets] = createBrickMesh(bounds, elementCounts)
%CREATEBRICKMESH Create a conforming structured tetrahedral mesh.

nx = elementCounts(1);
ny = elementCounts(2);
nz = elementCounts(3);

x = linspace(bounds(1, 1), bounds(2, 1), nx + 1);
y = linspace(bounds(1, 2), bounds(2, 2), ny + 1);
z = linspace(bounds(1, 3), bounds(2, 3), nz + 1);
[X, Y, Z] = ndgrid(x, y, z);
nodes = [X(:), Y(:), Z(:)];

nodeIndex = @(i, j, k) i + (nx + 1) * (j - 1) + ...
    (nx + 1) * (ny + 1) * (k - 1);
tets = zeros(6 * nx * ny * nz, 4);
tetIndex = 0;

for k = 1:nz
    for j = 1:ny
        for i = 1:nx
            n000 = nodeIndex(i,     j,     k);
            n100 = nodeIndex(i + 1, j,     k);
            n010 = nodeIndex(i,     j + 1, k);
            n110 = nodeIndex(i + 1, j + 1, k);
            n001 = nodeIndex(i,     j,     k + 1);
            n101 = nodeIndex(i + 1, j,     k + 1);
            n011 = nodeIndex(i,     j + 1, k + 1);
            n111 = nodeIndex(i + 1, j + 1, k + 1);

            cellTets = [n000, n100, n110, n111;
                        n000, n110, n010, n111;
                        n000, n010, n011, n111;
                        n000, n011, n001, n111;
                        n000, n001, n101, n111;
                        n000, n101, n100, n111];
            tets(tetIndex + (1:6), :) = cellTets;
            tetIndex = tetIndex + 6;
        end
    end
end
end


function [K, M] = assembleLinearElasticity(nodes, tets, E, nu, rho)
%ASSEMBLELINEARELASTICITY Assemble 3-D linear tetrahedral FEM matrices.

nodeCount = size(nodes, 1);
dofCount = 3 * nodeCount;
tetCount = size(tets, 1);
entryCount = 12 * 12 * tetCount;
rowIndices = zeros(entryCount, 1);
columnIndices = zeros(entryCount, 1);
stiffnessValues = zeros(entryCount, 1);
massValues = zeros(entryCount, 1);

lameLambda = E * nu / ((1 + nu) * (1 - 2 * nu));
shearModulus = E / (2 * (1 + nu));
D = [lameLambda + 2 * shearModulus, lameLambda, lameLambda, 0, 0, 0;
     lameLambda, lameLambda + 2 * shearModulus, lameLambda, 0, 0, 0;
     lameLambda, lameLambda, lameLambda + 2 * shearModulus, 0, 0, 0;
     0, 0, 0, shearModulus, 0, 0;
     0, 0, 0, 0, shearModulus, 0;
     0, 0, 0, 0, 0, shearModulus];

offset = 0;
for elementIndex = 1:tetCount
    connectivity = tets(elementIndex, :);
    coordinates = nodes(connectivity, :);
    jacobian = [coordinates(2, :) - coordinates(1, :);
                coordinates(3, :) - coordinates(1, :);
                coordinates(4, :) - coordinates(1, :)];
    volume = abs(det(jacobian)) / 6;
    if volume <= eps
        error('The generated mesh contains a degenerate tetrahedron.');
    end

    shapeCoefficients = [ones(4, 1), coordinates] \ eye(4);
    gradients = shapeCoefficients(2:4, :);
    B = zeros(6, 12);
    for localNode = 1:4
        column = 3 * localNode - 2;
        gx = gradients(1, localNode);
        gy = gradients(2, localNode);
        gz = gradients(3, localNode);
        B(1, column) = gx;
        B(2, column + 1) = gy;
        B(3, column + 2) = gz;
        B(4, column) = gy;
        B(4, column + 1) = gx;
        B(5, column + 1) = gz;
        B(5, column + 2) = gy;
        B(6, column) = gz;
        B(6, column + 2) = gx;
    end

    elementK = volume * (B.' * D * B);
    scalarMass = (rho * volume / 20) * (ones(4) + eye(4));
    elementM = kron(scalarMass, eye(3));

    % The exported matrices use component-block ordering: all x DOFs, then
    % all y DOFs, then all z DOFs.  Local element matrices are interleaved.
    elementDofs = reshape([connectivity;
                           connectivity + nodeCount;
                           connectivity + 2 * nodeCount], [], 1);
    [elementRows, elementColumns] = ndgrid(elementDofs, elementDofs);
    range = offset + (1:144);
    rowIndices(range) = elementRows(:);
    columnIndices(range) = elementColumns(:);
    stiffnessValues(range) = elementK(:);
    massValues(range) = elementM(:);
    offset = offset + 144;
end

K = sparse(rowIndices, columnIndices, stiffnessValues, dofCount, dofCount);
M = sparse(rowIndices, columnIndices, massValues, dofCount, dofCount);
K = 0.5 * (K + K.');
M = 0.5 * (M + M.');
end


function [KReduced, MReduced, T] = ...
        craigBamptonReduction(K, M, nodes, origins, fixedInterfaceModeCount)
%CRAIGBAMPTONREDUCTION Reduce to interface and fixed-interface coordinates.

nodeCount = size(nodes, 1);
dofCount = 3 * nodeCount;
frameCount = size(origins, 1);
interfaceDofCount = 6 * frameCount;
tolerance = 1e-10 * max(1, max(max(nodes, [], 1) - min(nodes, [], 1)));

boundaryNodes = [];
frameForNode = [];
for frameIndex = 1:frameCount
    nodesOnFrame = find(abs(nodes(:, 1) - origins(frameIndex, 1)) <= tolerance);
    boundaryNodes = [boundaryNodes; nodesOnFrame]; %#ok<AGROW>
    frameForNode = [frameForNode; frameIndex * ones(numel(nodesOnFrame), 1)]; %#ok<AGROW>
end

if numel(unique(boundaryNodes)) ~= numel(boundaryNodes)
    error('The Craig-Bampton interface node sets overlap.');
end

boundaryDofs = reshape([boundaryNodes, ...
                        boundaryNodes + nodeCount, ...
                        boundaryNodes + 2 * nodeCount].', [], 1);
interiorDofs = setdiff((1:dofCount).', boundaryDofs);
G = zeros(numel(boundaryDofs), interfaceDofCount);

for boundaryIndex = 1:numel(boundaryNodes)
    frameIndex = frameForNode(boundaryIndex);
    relativePosition = nodes(boundaryNodes(boundaryIndex), :) - origins(frameIndex, :);
    rigidMap = [eye(3), -skewMatrix(relativePosition)];
    rows = 3 * boundaryIndex - 2:3 * boundaryIndex;
    columns = 6 * frameIndex - 5:6 * frameIndex;
    G(rows, columns) = rigidMap;
end

Kii = K(interiorDofs, interiorDofs);
Kib = K(interiorDofs, boundaryDofs);
constraintModes = -(Kii \ (Kib * G));

[fixedInterfaceModes, eigenvalueMatrix] = eig(full(Kii), full(M(interiorDofs, interiorDofs)));
eigenvalues = real(diag(eigenvalueMatrix));
[eigenvalues, order] = sort(eigenvalues); %#ok<ASGLU>
fixedInterfaceModes = real(fixedInterfaceModes(:, order));

if fixedInterfaceModeCount < 0 || ...
        fixedInterfaceModeCount ~= floor(fixedInterfaceModeCount) || ...
        fixedInterfaceModeCount > size(fixedInterfaceModes, 2)
    error('fixedInterfaceModeCount must be an integer in [0, %d].', ...
        size(fixedInterfaceModes, 2));
end

fixedInterfaceModes = fixedInterfaceModes(:, 1:fixedInterfaceModeCount);
Mii = M(interiorDofs, interiorDofs);
for modeIndex = 1:fixedInterfaceModeCount
    modalMass = fixedInterfaceModes(:, modeIndex).' * ...
        Mii * fixedInterfaceModes(:, modeIndex);
    fixedInterfaceModes(:, modeIndex) = ...
        fixedInterfaceModes(:, modeIndex) / sqrt(modalMass);

    % Remove the arbitrary eigensolver sign so repeated exports are stable.
    [~, pivot] = max(abs(fixedInterfaceModes(:, modeIndex)));
    if fixedInterfaceModes(pivot, modeIndex) < 0
        fixedInterfaceModes(:, modeIndex) = ...
            -fixedInterfaceModes(:, modeIndex);
    end
end

T = zeros(dofCount, interfaceDofCount + fixedInterfaceModeCount);
T(boundaryDofs, 1:interfaceDofCount) = G;
T(interiorDofs, 1:interfaceDofCount) = constraintModes;
T(interiorDofs, interfaceDofCount + 1:end) = fixedInterfaceModes;

KReduced = full(T.' * K * T);
MReduced = full(T.' * M * T);
KReduced = 0.5 * (KReduced + KReduced.');
MReduced = 0.5 * (MReduced + MReduced.');
end


function C = modalDampingMatrix(zeta, K, M)
%MODALDAMPINGMATRIX Construct a constant-ratio modal damping matrix.

[vectors, eigenvalueMatrix] = eig(0.5 * (K + K.'), 0.5 * (M + M.'));
eigenvalues = real(diag(eigenvalueMatrix));
[eigenvalues, order] = sort(eigenvalues);
vectors = real(vectors(:, order));

largestEigenvalue = max(1, max(abs(eigenvalues)));
eigenvalues(eigenvalues < 1e-10 * largestEigenvalue) = 0;
frequencies = sqrt(max(eigenvalues, 0));

for modeIndex = 1:size(vectors, 2)
    modalMass = vectors(:, modeIndex).' * M * vectors(:, modeIndex);
    vectors(:, modeIndex) = vectors(:, modeIndex) / sqrt(modalMass);
end

C = M * vectors * diag(2 * zeta * frequencies) * vectors.' * M;
C = real(0.5 * (C + C.'));
end


function S = skewMatrix(vector)
%SKEWMATRIX Return the matrix whose product is vector cross another vector.

S = [0, -vector(3), vector(2);
     vector(3), 0, -vector(1);
     -vector(2), vector(1), 0];
end


function [F, surfaceNodeIds] = extractSurface(tets, nodes)
%EXTRACTSURFACE Extract consistently outward-facing boundary triangles.

tetCount = size(tets, 1);
allFaces = zeros(4 * tetCount, 3);

for tetIndex = 1:tetCount
    tet = tets(tetIndex, :);
    localFaces = [tet([1, 2, 3]);
                  tet([1, 4, 2]);
                  tet([2, 4, 3]);
                  tet([3, 4, 1])];
    oppositeNodes = tet([4, 3, 1, 2]);

    for localFaceIndex = 1:4
        face = localFaces(localFaceIndex, :);
        a = nodes(face(1), :);
        b = nodes(face(2), :);
        c = nodes(face(3), :);
        opposite = nodes(oppositeNodes(localFaceIndex), :);
        if dot(cross(b - a, c - a), opposite - a) > 0
            face([2, 3]) = face([3, 2]);
        end
        allFaces(4 * tetIndex - 4 + localFaceIndex, :) = face;
    end
end

[~, ~, faceGroup] = unique(sort(allFaces, 2), 'rows');
faceCounts = accumarray(faceGroup, 1);
boundaryRows = faceCounts(faceGroup) == 1;
boundaryFaces = allFaces(boundaryRows, :);

surfaceNodeIds = unique(boundaryFaces(:));
surfaceIndex = zeros(size(nodes, 1), 1);
surfaceIndex(surfaceNodeIds) = 1:numel(surfaceNodeIds);
F = uint32(surfaceIndex(boundaryFaces));
end


function writeCsv(filename, data)
%WRITECSV Write a numeric matrix as comma-separated values without a header.

dlmwrite(filename, full(data), ',', 'precision', '%.17g');
end
