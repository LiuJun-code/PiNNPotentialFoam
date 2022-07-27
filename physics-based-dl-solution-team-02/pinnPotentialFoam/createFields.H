    #include "volFieldsFwd.H"

/*
    volScalarField vf 
    (
        IOobject
        (
            "vf",
            runTime.timeName(), 
            mesh, 
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
    volScalarField vf_nn("vf_nn", vf);
    
    volScalarField error_c 
    (
        IOobject
        (
            "error_c", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedScalar("error_c", vf.dimensions(), 0)
    );
    
    volVectorField gradvf_nn
    (
        IOobject
        (
            "gradvf_nn", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedVector("gradvf_nn", vf.dimensions(), vector(0,0,0))
    );
*/

    // XXX: Read Phi field. Guoliang
    volScalarField Phi
    (
        IOobject
        (
            "Phi",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
    volScalarField Phi_nn("Phi_nn", Phi);

    volScalarField error_Phi 
    (
        IOobject
        (
            "error_Phi", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedScalar("error_Phi", Phi.dimensions(), 0)
    );
    
    volVectorField lapPhi_nn
    (
        IOobject
        (
            "lapPhi_nn", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedVector("lapPhi_nn", Phi.dimensions(), vector(0,0,0))
    );

    // XXX: Read U field. Guoliang
    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
    volVectorField U_nn("U_nn", U);
    
    volScalarField error_U 
    (
        IOobject
        (
            "error_U", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedScalar("error_U", U.dimensions(), 0)
    );
    
    volTensorField gradU_nn
    (
        IOobject
        (
            "gradU_nn", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedTensor("gradU_nn", U.dimensions(), tensor::zero)
    );

