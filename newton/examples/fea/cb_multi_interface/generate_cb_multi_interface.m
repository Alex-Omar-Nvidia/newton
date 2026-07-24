function generate_cb_multi_interface()
%GENERATE_CB_MULTI_INTERFACE Export and verify a three-interface CB model.
%
% The component is an extruded T-bracket with rigid six-DOF interfaces on
% the left, right, and top arm ends.  The script intentionally uses only
% core MATLAB/Octave functionality: it creates a structured tetrahedral
% mesh, assembles first-order linear tetrahedral FEM matrices, performs a
% Craig-Bampton reduction, and advances a dense reduced reference system.
%
% The primary reference rotates a finite rigid frame about the top interface's
% local +Z axis while integrating small elastic motion in the co-rotating
% frame.  A secondary self-equilibrated force-pulse reference exercises all
% three ports directly.
%
% Run from MATLAB or Octave:
%
%   octave --quiet generate_cb_multi_interface.m
%
% The checked-in CSV files let the Newton example run without MATLAB or
% Octave.  Re-run this script whenever the mesh, material, reduction, or
% reference motion or load changes.

format long

scriptDir = fileparts(mfilename('fullpath'));

youngsModulus = 1.0e6;       % [Pa]
poissonsRatio = 0.30;
density = 1200.0;            % [kg/m^3]
dampingRatio = 0.02;
fixedInterfaceModeCount = 6;

[nodes, tets] = createTBracketMesh();
[stiffness, mass] = assembleLinearElasticity( ...
    nodes, tets, youngsModulus, poissonsRatio, density);

interfaceOrigins = [-0.6, 0.0, 0.0;
                     0.6, 0.0, 0.0;
                     0.0, 0.0, 0.6];
tolerance = 1.0e-10;
interfaceNodeIds = {
    find(abs(nodes(:, 1) - interfaceOrigins(1, 1)) <= tolerance);
    find(abs(nodes(:, 1) - interfaceOrigins(2, 1)) <= tolerance);
    find(abs(nodes(:, 3) - interfaceOrigins(3, 3)) <= tolerance)
};

[reducedStiffness, reducedMass, transform] = craigBamptonReduction( ...
    stiffness, mass, nodes, interfaceOrigins, interfaceNodeIds, ...
    fixedInterfaceModeCount);
reducedDamping = modalDampingMatrix( ...
    dampingRatio, reducedStiffness, reducedMass);

[faces, surfaceNodeIds] = extractSurface(tets, nodes);
interiorNodeIds = setdiff((1:size(nodes, 1)).', surfaceNodeIds);
sampleNodeIds = [surfaceNodeIds; interiorNodeIds];
samplePoints = nodes(sampleNodeIds, :);
recovery = sampleRecovery(transform, sampleNodeIds, size(nodes, 1));

writeCsv(fullfile(scriptDir, 'cb_rom_stiffness.csv'), reducedStiffness);
writeCsv(fullfile(scriptDir, 'cb_rom_mass.csv'), reducedMass);
writeCsv(fullfile(scriptDir, 'cb_rom_damping.csv'), reducedDamping);
writeCsv(fullfile(scriptDir, 'cb_rom_interfaces.csv'), interfaceOrigins);
writeCsv(fullfile(scriptDir, 'cb_rom_recovery.csv'), recovery);
writeCsv(fullfile(scriptDir, 'cb_rom_points.csv'), samplePoints);
writeCsv(fullfile(scriptDir, 'cb_rom_faces.csv'), faces);

referenceTimeStep = 2.0e-4;   % [s]
referenceStepCount = 240;
pulseDuration = 4.0e-3;       % [s]
peakPortForce = 100.0;        % [N]
balancedForce = zeros(size(reducedMass, 1), 1);
balancedForce(3) = peakPortForce;
balancedForce(9) = peakPortForce;
balancedForce(15) = -2.0 * peakPortForce;

rigidMap = reducedRigidMap(interfaceOrigins, size(reducedMass, 1));
equilibriumResidual = norm(rigidMap.' * balancedForce);
if equilibriumResidual > 1.0e-10 * peakPortForce
    error('The three-interface reference force is not self-equilibrated.');
end

probePoints = [-0.6,  0.0, 0.0;
                0.6,  0.0, 0.0;
                0.0,  0.0, 0.6;
                0.0, -0.1, 0.1];
probeRows = recoveryRowsForPoints(samplePoints, probePoints);
[referenceState, referenceProbes] = integrateReference( ...
    reducedMass, reducedStiffness, reducedDamping, recovery, probeRows, ...
    balancedForce, referenceTimeStep, referenceStepCount, pulseDuration);

frequencies = elasticFrequencies(reducedStiffness, reducedMass);
metadata = [referenceTimeStep, referenceStepCount, pulseDuration, ...
            peakPortForce, fixedInterfaceModeCount];
writeCsv(fullfile(scriptDir, 'reference_metadata.csv'), metadata);
writeCsv(fullfile(scriptDir, 'reference_balanced_interface_force.csv'), ...
    balancedForce);
writeCsv(fullfile(scriptDir, 'reference_probe_points.csv'), probePoints);
writeCsv(fullfile(scriptDir, 'reference_reduced_state.csv'), referenceState);
writeCsv(fullfile(scriptDir, 'reference_probe_displacement.csv'), ...
    referenceProbes);
writeCsv(fullfile(scriptDir, 'reference_frequencies_hz.csv'), frequencies);

rotationTimeStep = 1.0 / 300.0;  % [s]
rotationPeriod = 4.0;            % [s]
rotationCycleCount = 1;
rotationStepCount = round(rotationCycleCount * rotationPeriod / ...
    rotationTimeStep);
rotationAmplitude = pi;          % [rad]
topInterfaceIndex = 3;
rotationJointStiffness = 1.0e6;
[rotationState, rotationProbes, rotationFrequencies] = ...
    integrateRotatingTopReference( ...
        reducedMass, reducedStiffness, reducedDamping, rigidMap, ...
        recovery, probeRows, rotationTimeStep, rotationStepCount, ...
        rotationPeriod, rotationAmplitude, topInterfaceIndex, ...
        rotationJointStiffness);
rotationMetadata = [rotationTimeStep, rotationStepCount, rotationPeriod, ...
                    rotationAmplitude, topInterfaceIndex, ...
                    rotationJointStiffness];
writeCsv(fullfile(scriptDir, 'rotation_reference_metadata.csv'), ...
    rotationMetadata);
writeCsv(fullfile(scriptDir, 'rotation_reference_reduced_state.csv'), ...
    rotationState);
writeCsv(fullfile(scriptDir, ...
    'rotation_reference_probe_displacement.csv'), rotationProbes);
writeCsv(fullfile(scriptDir, 'rotation_reference_frequencies_hz.csv'), ...
    rotationFrequencies);

fprintf('Wrote a %d-node, %d-tetrahedron T-bracket model.\n', ...
    size(nodes, 1), size(tets, 1));
fprintf(['Wrote a %d-coordinate Craig-Bampton model with three interfaces ', ...
         'and %d fixed-interface modes.\n'], ...
    size(reducedMass, 1), fixedInterfaceModeCount);
fprintf('Reference force rigid-equilibrium residual: %.3e.\n', ...
    equilibriumResidual);
fprintf(['Wrote a %d-step top-axis rotation reference spanning %d cycles ', ...
         'between -180 and +180 degrees.\n'], ...
    rotationStepCount, rotationCycleCount);
end


function [nodes, tets] = createTBracketMesh()
%CREATETBRACKETMESH Create an extruded, conforming T-bracket tetrahedral mesh.

x = -0.6:0.2:0.6;
y = -0.1:0.1:0.1;
z = -0.1:0.1:0.6;
[X, Y, Z] = ndgrid(x, y, z);
fullNodes = [X(:), Y(:), Z(:)];
nx = numel(x) - 1;
ny = numel(y) - 1;
nz = numel(z) - 1;

nodeIndex = @(i, j, k) i + (nx + 1) * (j - 1) + ...
    (nx + 1) * (ny + 1) * (k - 1);
fullTets = zeros(6 * nx * ny * nz, 4);
tetCount = 0;

for k = 1:nz
    zCenter = 0.5 * (z(k) + z(k + 1));
    for j = 1:ny
        for i = 1:nx
            xCenter = 0.5 * (x(i) + x(i + 1));
            inCrossbar = zCenter <= 0.1 + eps;
            inStem = abs(xCenter) <= 0.2 + eps && zCenter >= 0.1 - eps;
            if ~(inCrossbar || inStem)
                continue
            end

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
            fullTets(tetCount + (1:6), :) = cellTets;
            tetCount = tetCount + 6;
        end
    end
end

fullTets = fullTets(1:tetCount, :);
usedNodes = unique(fullTets(:));
nodeMap = zeros(size(fullNodes, 1), 1);
nodeMap(usedNodes) = 1:numel(usedNodes);
nodes = fullNodes(usedNodes, :);
tets = nodeMap(fullTets);
end


function [K, M] = assembleLinearElasticity(nodes, tets, E, nu, rho)
%ASSEMBLELINEARELASTICITY Assemble 3-D first-order tetrahedral FEM matrices.

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


function [KReduced, MReduced, T] = craigBamptonReduction( ...
        K, M, nodes, origins, interfaceNodeIds, fixedInterfaceModeCount)
%CRAIGBAMPTONREDUCTION Reduce three rigid interfaces and the interior.

nodeCount = size(nodes, 1);
dofCount = 3 * nodeCount;
frameCount = size(origins, 1);
interfaceDofCount = 6 * frameCount;
boundaryNodes = [];
frameForNode = [];

for frameIndex = 1:frameCount
    ids = interfaceNodeIds{frameIndex};
    if numel(ids) < 3
        error('Each rigid interface needs at least three mesh nodes.');
    end
    boundaryNodes = [boundaryNodes; ids(:)]; %#ok<AGROW>
    frameForNode = [frameForNode; ...
        frameIndex * ones(numel(ids), 1)]; %#ok<AGROW>
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
    relativePosition = nodes(boundaryNodes(boundaryIndex), :) - ...
        origins(frameIndex, :);
    rigidMap = [eye(3), -skewMatrix(relativePosition)];
    rows = 3 * boundaryIndex - 2:3 * boundaryIndex;
    columns = 6 * frameIndex - 5:6 * frameIndex;
    G(rows, columns) = rigidMap;
end

Kii = K(interiorDofs, interiorDofs);
Kib = K(interiorDofs, boundaryDofs);
constraintModes = -(Kii \ (Kib * G));

[fixedModes, eigenvalueMatrix] = eig( ...
    full(Kii), full(M(interiorDofs, interiorDofs)));
eigenvalues = real(diag(eigenvalueMatrix));
[eigenvalues, order] = sort(eigenvalues); %#ok<ASGLU>
fixedModes = real(fixedModes(:, order));
fixedModes = fixedModes(:, 1:fixedInterfaceModeCount);

Mii = M(interiorDofs, interiorDofs);
for modeIndex = 1:fixedInterfaceModeCount
    modalMass = fixedModes(:, modeIndex).' * Mii * fixedModes(:, modeIndex);
    fixedModes(:, modeIndex) = fixedModes(:, modeIndex) / sqrt(modalMass);
    [~, pivot] = max(abs(fixedModes(:, modeIndex)));
    if fixedModes(pivot, modeIndex) < 0
        fixedModes(:, modeIndex) = -fixedModes(:, modeIndex);
    end
end

T = zeros(dofCount, interfaceDofCount + fixedInterfaceModeCount);
T(boundaryDofs, 1:interfaceDofCount) = G;
T(interiorDofs, 1:interfaceDofCount) = constraintModes;
T(interiorDofs, interfaceDofCount + 1:end) = fixedModes;

KReduced = full(T.' * K * T);
MReduced = full(T.' * M * T);
KReduced = 0.5 * (KReduced + KReduced.');
MReduced = 0.5 * (MReduced + MReduced.');
end


function recovery = sampleRecovery(T, sampleNodeIds, nodeCount)
%SAMPLERECOVERY Map reduced coordinates to sample-node displacement.

reducedDofCount = size(T, 2);
recovery = zeros(3 * numel(sampleNodeIds), reducedDofCount);
for sample = 1:numel(sampleNodeIds)
    node = sampleNodeIds(sample);
    recovery(3 * sample - 2, :) = T(node, :);
    recovery(3 * sample - 1, :) = T(node + nodeCount, :);
    recovery(3 * sample, :) = T(node + 2 * nodeCount, :);
end
end


function rigid = reducedRigidMap(origins, reducedDofCount)
%REDUCEDRIGIDMAP Map a global rigid twist to all interface coordinates.

rigid = zeros(reducedDofCount, 6);
for interface = 1:size(origins, 1)
    first = 6 * interface - 5;
    rigid(first:first + 2, 1:3) = eye(3);
    rigid(first:first + 2, 4:6) = -skewMatrix(origins(interface, :));
    rigid(first + 3:first + 5, 4:6) = eye(3);
end
end


function rows = recoveryRowsForPoints(surfacePoints, probePoints)
%RECOVERYROWSFORPOINTS Find interleaved recovery rows for named probes.

rows = zeros(size(probePoints, 1), 3);
for probe = 1:size(probePoints, 1)
    distances = sqrt(sum((surfacePoints - probePoints(probe, :)).^2, 2));
    [distance, sample] = min(distances);
    if distance > 1.0e-10
        error('Reference probe is not a surface sample.');
    end
    rows(probe, :) = 3 * sample - 2:3 * sample;
end
end


function [stateHistory, probeHistory] = integrateReference( ...
        M, K, C, recovery, probeRows, forceVector, dt, stepCount, pulseDuration)
%INTEGRATEREFERENCE Advance the dense CB equations with implicit Euler.

reducedDofCount = size(M, 1);
probeCount = size(probeRows, 1);
q = zeros(reducedDofCount, 1);
v = zeros(reducedDofCount, 1);
systemMatrix = M + dt * C + dt * dt * K;
stateHistory = zeros(stepCount + 1, 2 + 2 * reducedDofCount);
probeHistory = zeros(stepCount + 1, 2 + 3 * probeCount);

for row = 1:stepCount + 1
    time = (row - 1) * dt;
    loadScale = referenceLoadScale(time, pulseDuration);
    displacement = recovery * q;
    probeDisplacement = zeros(1, 3 * probeCount);
    for probe = 1:probeCount
        columns = 3 * probe - 2:3 * probe;
        probeDisplacement(columns) = displacement(probeRows(probe, :)).';
    end
    stateHistory(row, :) = [time, loadScale, q.', v.'];
    probeHistory(row, :) = [time, loadScale, probeDisplacement];

    if row <= stepCount
        nextTime = row * dt;
        nextScale = referenceLoadScale(nextTime, pulseDuration);
        appliedForce = nextScale * forceVector;
        v = systemMatrix \ (M * v + dt * (appliedForce - K * q));
        q = q + dt * v;
    end
end
end


function scale = referenceLoadScale(time, pulseDuration)
%REFERENCELOADSCALE Half-sine force pulse.

if time <= 0.0 || time >= pulseDuration
    scale = 0.0;
else
    scale = sin(pi * time / pulseDuration);
end
end


function [stateHistory, probeHistory, frequencies] = ...
        integrateRotatingTopReference( ...
            M, K, C, rigidMap, recovery, probeRows, dt, stepCount, ...
            period, amplitude, topInterfaceIndex, jointStiffness)
%INTEGRATEROTATINGTOPREFERENCE Solve top-fixture base excitation.
%
% The finite rotation belongs to a rigid frame and is not inserted into the
% linear Craig-Bampton coordinates.  In that co-rotating frame, a six-DOF
% penalty fixture couples the top interface to the drive.  Its angular
% acceleration about local +Z produces the standard base-excitation force
% -M * rigid_rotation_z * alpha.

reducedDofCount = size(M, 1);
topDofs = 6 * topInterfaceIndex - 5:6 * topInterfaceIndex;
fixtureStiffness = zeros(reducedDofCount, reducedDofCount);
fixtureStiffness(topDofs, topDofs) = jointStiffness * eye(6);
supportedStiffness = K + fixtureStiffness;
rigidRotationZ = rigidMap(:, 6);
baseExcitation = -M * rigidRotationZ;
systemMatrix = M + dt * C + dt * dt * supportedStiffness;

relativeQ = zeros(reducedDofCount, 1);
relativeV = zeros(reducedDofCount, 1);
probeCount = size(probeRows, 1);
stateHistory = zeros(stepCount + 1, 4 + 2 * reducedDofCount);
probeHistory = zeros(stepCount + 1, 4 + 3 * probeCount);

for row = 1:stepCount + 1
    time = (row - 1) * dt;
    [angle, angularSpeed, angularAcceleration] = ...
        rotationKinematics(time, period, amplitude);
    displacement = recovery * relativeQ;
    probeDisplacement = zeros(1, 3 * probeCount);
    for probe = 1:probeCount
        columns = 3 * probe - 2:3 * probe;
        probeDisplacement(columns) = displacement(probeRows(probe, :)).';
    end
    stateHistory(row, :) = [time, angle, angularSpeed, ...
                            angularAcceleration, relativeQ.', relativeV.'];
    probeHistory(row, :) = [time, angle, angularSpeed, ...
                            angularAcceleration, probeDisplacement];

    if row <= stepCount
        nextTime = row * dt;
        [~, ~, nextAngularAcceleration] = ...
            rotationKinematics(nextTime, period, amplitude);
        appliedForce = baseExcitation * nextAngularAcceleration;
        relativeV = systemMatrix \ ( ...
            M * relativeV + ...
            dt * (appliedForce - supportedStiffness * relativeQ));
        relativeQ = relativeQ + dt * relativeV;
    end
end

eigenvalues = eig(0.5 * (supportedStiffness + supportedStiffness.'), ...
                  0.5 * (M + M.'));
eigenvalues = sort(real(eigenvalues));
frequencies = sqrt(max(eigenvalues, 0.0)) / (2.0 * pi);
end


function [angle, angularSpeed, angularAcceleration] = ...
        rotationKinematics(time, period, amplitude)
%ROTATIONKINEMATICS Smooth periodic motion between +/- amplitude.

frequency = 2.0 * pi / period;
angle = amplitude * sin(frequency * time);
angularSpeed = amplitude * frequency * cos(frequency * time);
angularAcceleration = -amplitude * frequency * frequency * ...
    sin(frequency * time);
end


function frequencies = elasticFrequencies(K, M)
%ELASTICFREQUENCIES Return non-rigid generalized eigenfrequencies [Hz].

[~, eigenvalueMatrix] = eig(0.5 * (K + K.'), 0.5 * (M + M.'));
eigenvalues = sort(real(diag(eigenvalueMatrix)));
scale = max(1.0, max(abs(eigenvalues)));
eigenvalues = eigenvalues(eigenvalues > 1.0e-8 * scale);
frequencies = sqrt(eigenvalues) / (2.0 * pi);
end


function C = modalDampingMatrix(zeta, K, M)
%MODALDAMPINGMATRIX Construct constant-ratio classical damping.

[vectors, eigenvalueMatrix] = eig(0.5 * (K + K.'), 0.5 * (M + M.'));
eigenvalues = real(diag(eigenvalueMatrix));
[eigenvalues, order] = sort(eigenvalues);
vectors = real(vectors(:, order));
largestEigenvalue = max(1.0, max(abs(eigenvalues)));
eigenvalues(eigenvalues < 1.0e-10 * largestEigenvalue) = 0.0;
frequencies = sqrt(max(eigenvalues, 0.0));

for modeIndex = 1:size(vectors, 2)
    modalMass = vectors(:, modeIndex).' * M * vectors(:, modeIndex);
    vectors(:, modeIndex) = vectors(:, modeIndex) / sqrt(modalMass);
end
C = M * vectors * diag(2.0 * zeta * frequencies) * vectors.' * M;
C = real(0.5 * (C + C.'));
end


function S = skewMatrix(vector)
%SKEWMATRIX Return the cross-product matrix of a three-vector.

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
    for localFace = 1:4
        face = localFaces(localFace, :);
        a = nodes(face(1), :);
        b = nodes(face(2), :);
        c = nodes(face(3), :);
        opposite = nodes(oppositeNodes(localFace), :);
        if dot(cross(b - a, c - a), opposite - a) > 0
            face([2, 3]) = face([3, 2]);
        end
        allFaces(4 * tetIndex - 4 + localFace, :) = face;
    end
end

[~, ~, faceGroup] = unique(sort(allFaces, 2), 'rows');
faceCounts = accumarray(faceGroup, 1);
boundaryFaces = allFaces(faceCounts(faceGroup) == 1, :);
surfaceNodeIds = unique(boundaryFaces(:));
surfaceIndex = zeros(size(nodes, 1), 1);
surfaceIndex(surfaceNodeIds) = 1:numel(surfaceNodeIds);
F = uint32(surfaceIndex(boundaryFaces));
end


function writeCsv(filename, data)
%WRITECSV Write a headerless CSV matrix with stable precision.

dlmwrite(filename, full(data), ',', 'precision', '%.17g');
end
