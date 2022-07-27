/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2022 Tomislav Maric, TU Darmstadt 
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    pinnPotentialFoam

Description

\*---------------------------------------------------------------------------*/

// libtorch
#include <torch/torch.h>
#include "ATen/Functions.h"
#include "ATen/core/interned_strings.h"
#include "torch/nn/modules/activation.h"
#include "torch/optim/lbfgs.h"
#include "torch/optim/rmsprop.h"

// STL 
#include <algorithm>
#include <random> 
#include <numeric>
#include <cmath>
#include <filesystem>

// OpenFOAM 
#include "fvCFD.H"

// libtorch-OpenFOAM data transfer
#include "torchFunctions.C"
#include "fileNameGenerator.H"

using namespace Foam;
using namespace torch::indexing;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addOption
    (
        "volFieldName",
        "string",
        "Name of the volume (cell-centered) field approximated by the neural network."
    );

    argList::addOption
    (
        "hiddenLayers",
        "int,int,int,...",
        "A sequence of hidden-layer depths."
    );

    argList::addOption
    (
        "optimizerStep",
        "double",
        "Step of the optimizer."
    );

    argList::addOption
    (
        "maxIterations",
        "<int>",
        "Max number of iterations."
    );
    
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    
    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    // Initialize hyperparameters 
    
    // - NN architecture 
    DynamicList<label> hiddenLayers;
    scalar optimizerStep;
    // - Maximal number of training iterations.
    std::size_t maxIterations;
    
    // - Initialize hyperparameters from command line arguments if they are provided
    if (args.found("hiddenLayers") && 
        args.found("optimizerStep") &&
        args.found("maxIterations"))
    {
        hiddenLayers = args.get<DynamicList<label>>("hiddenLayers");
        optimizerStep = args.get<scalar>("optimizerStep");
        maxIterations = args.get<label>("maxIterations");
    } 
    else // Initialize from system/fvSolution.AI.approximator sub-dict.
    {
        const fvSolution& fvSolutionDict (mesh);
        const dictionary& aiDict = fvSolutionDict.subDict("AI");

        hiddenLayers = aiDict.get<DynamicList<label>>("hiddenLayers");
        optimizerStep = aiDict.get<scalar>("optimizerStep");
        maxIterations = aiDict.get<label>("maxIterations");
    }
    
    // Use double-precision floating-point arithmetic. 
    torch::set_default_dtype(
        torch::scalarTypeToTypeMeta(torch::kDouble)
    );
    
    // Construct the MLP 
    torch::nn::Sequential nn;
    // - Input layer are always the 3 spatial coordinates in OpenFOAM, 2D
    //   simulations are pseudo-2D (single cell-layer).
    nn->push_back(torch::nn::Linear(3, hiddenLayers[0])); 
    nn->push_back(torch::nn::GELU()); // FIXME: RTS activation function.
    // - Hidden layers
    for (label L=1; L < hiddenLayers.size(); ++L)
    {
        nn->push_back(
            torch::nn::Linear(hiddenLayers[L-1], hiddenLayers[L])
        );
        // TODO: RTS Alternatives TM.
        nn->push_back(torch::nn::GELU()); 
        //nn->push_back(torch::nn::Tanh()); 
    }
    // - Output is 1D: value of the learned scalar field. 
    // TODO: generalize here for vector / scalar data. 
    nn->push_back(
        torch::nn::Linear(hiddenLayers[hiddenLayers.size() - 1], 4)
    );
    
    // Initialize training data 
    
    // - Reinterpreting OpenFOAM's fileds as torch::tensors without copying
    // - Reinterpret OpenFOAM's input volScalarField as scalar* array 
    volScalarField::pointer Phi_data = Phi.ref().data();
    volVectorField::pointer U_data = U.ref().data();

    // - Use the scalar* (volScalarField::pointer) to view 
    //   the volScalarField as torch::Tensor without copying data. 
    torch::Tensor Phi_tensor = torch::from_blob(Phi_data, {Phi.size(), 1});
    torch::Tensor U_tensor = torch::from_blob(U_data, {U.size(), 3});
    torch::Tensor PhiU_tensor = torch::cat({Phi_tensor, U_tensor}, 1);

    // - Reinterpret OpenFOAM's vectorField as vector* array 
    volVectorField& cc = const_cast<volVectorField&>(mesh.C());
    volVectorField::pointer cc_data = cc.ref().data();
    // - Use the scalar* (volScalarField::pointer) to view 
    //   the volScalarField as torch::Tensor without copying data. 
    torch::Tensor cc_tensor = torch::from_blob(cc_data, {cc.size(),3});
    
    // - Randomly shuffle cell center indices. 
    torch::Tensor shuffled_indices = torch::randperm(
        mesh.nCells(),
        torch::TensorOptions().dtype(at::kLong)
    ); 
    // - Randomly select 10 % of all cell centers for training. 
    long int n_cells = int(0.1 * mesh.nCells());
    torch::Tensor training_indices = shuffled_indices.index({Slice(0, n_cells)}); 
    
    // - Use 10% of random indices to select the training_data from vf_tensor 
    torch::Tensor cc_training = cc_tensor.index(training_indices);
    cc_training.requires_grad_(true);
    torch::Tensor PhiU_training = PhiU_tensor.index(training_indices);
    PhiU_training.requires_grad_(true);
    
    // Train the network
    torch::optim::RMSprop optimizer(nn->parameters(), optimizerStep);

    torch::Tensor PhiU_predict = torch::zeros_like(PhiU_training);
    torch::Tensor mse = torch::zeros_like(PhiU_training);
    
    size_t epoch = 1;
    double min_mse = 10.; 

    // - Approximate DELTA_X on unstructured meshes
    const auto& deltaCoeffs = mesh.deltaCoeffs().internalField();
    double delta_x = Foam::pow(
        Foam::min(deltaCoeffs).value(),-1
    );
    
    // - Open the data file for writing
    auto file_name = getAvailableFileName("pinnPotentialFoam");   
    std::ofstream dataFile (file_name);
    dataFile << "HIDDEN_LAYERS,OPTIMIZER_STEP,MAX_ITERATIONS,"
        << "DELTA_X,EPOCH,DATA_MSE,RESD_MSE,TRAINING_MSE\n";

    volScalarField Phi_lap("Phi_lap", fvc::laplacian(Phi));
    volScalarField::pointer Phi_lap_data = Phi_lap.ref().data();
    torch::Tensor Phi_lap_tensor = torch::from_blob(Phi_lap_data, {Phi_lap.size(), 1});
    torch::Tensor Phi_lap_training = Phi_lap_tensor.index(training_indices);
    Phi_lap_training.requires_grad_(true);

    volScalarField U_div("U_div", fvc::div(U));
    volScalarField::pointer U_div_data = U_div.ref().data();
    torch::Tensor U_div_tensor = torch::from_blob(U_div_data, {U_div.size(), 1});
    torch::Tensor U_div_training = U_div_tensor.index(training_indices);
    U_div_training.requires_grad_(true);

    // - Initialize the best model (to be saved during training)
    torch::nn::Sequential nn_best;
    for (; epoch <= maxIterations; ++epoch) 
    {
        // Training
        optimizer.zero_grad();

        // Compute the prediction from the nn. 
        PhiU_predict = nn->forward(cc_training);

        // Compute the data mse loss.
        auto mse_data = mse_loss(PhiU_predict, PhiU_training);
        
        // Compute the gradient of the prediction w.r.t. input. --> d (vf_predict) / d (cc_training), its result
        // is a N_para * N_vftraining tensor
        auto U_0_tensor = PhiU_predict.index({Slice(),1});
        auto U_1_tensor = PhiU_predict.index({Slice(),2});
        auto U_2_tensor = PhiU_predict.index({Slice(),3});

        auto U_0_predict_grad = torch::autograd::grad(
           {U_0_tensor},  
           {cc_training}, 
           {torch::ones_like(U_0_tensor)}, 
           true  // create_graph = True ??
        );

        auto U_1_predict_grad = torch::autograd::grad(
           {U_1_tensor},  
           {cc_training}, 
           {torch::ones_like(U_1_tensor)}, 
           true  // create_graph = True ??
        );

            auto U_2_predict_grad = torch::autograd::grad(
           {U_2_tensor},  
           {cc_training},
           {torch::ones_like(U_2_tensor)},
           true  // create_graph = True ??
        );

       //  auto PhiU_predict_part = PhiU_predict.index({Slice(),Slice(None,1)});

        auto Phi_predict_grad = torch::autograd::grad(
           {PhiU_predict.index({Slice(),0})},  
           {cc_training}, 
           {torch::ones_like(U_2_tensor)},
           true,  // create_graph = True ??
           true
        );

        auto Phi_predict_grad_0 = Phi_predict_grad[0].index({Slice(),0});
        auto Phi_predict_grad_1 = Phi_predict_grad[0].index({Slice(),1});
        auto Phi_predict_grad_2 = Phi_predict_grad[0].index({Slice(),2});        

         auto Phi_predict_laplacian_0 = torch::autograd::grad(
           {Phi_predict_grad_0}, 
           {cc_training}, 
           {torch::ones_like(Phi_predict_grad_0)},
           true,
           true // create_graph = True ??
        );

         auto Phi_predict_laplacian_1 = torch::autograd::grad(
           {Phi_predict_grad_1}, 
           {cc_training},
           {torch::ones_like(Phi_predict_grad_1)},
           true,
           true // create_graph = True ??
        );

         auto Phi_predict_laplacian_2 = torch::autograd::grad(
           {Phi_predict_grad_2},  
           {cc_training}, 
           {torch::ones_like(Phi_predict_grad_2)},
           true,
           true // create_graph = True ??
        );

        auto Phi_predict_lap= Phi_predict_laplacian_0[0].index({Slice(),0})
                                    +Phi_predict_laplacian_1[0].index({Slice(),1})
                                    +Phi_predict_laplacian_2[0].index({Slice(),2});

        auto U_predict_div = U_0_predict_grad[0].index({Slice(),0})
                            + U_1_predict_grad[0].index({Slice(),1})
                            + U_2_predict_grad[0].index({Slice(),2});

        // - Compute the potential-flow residual mse. Guoliang
        auto mse_lap_div = mse_loss(
            Phi_predict_lap, //++check
            U_predict_div
        );

        /*
        auto mse_Phi_lap = mse_loss(
            Phi_predict_lap,
            Phi_lap_training.index({Slice(),0})
        );

        auto mse_U_div = mse_loss(
            U_predict_div,
            U_div_training.index({Slice(),0})
        );
        */

        auto mse_resd = mse_lap_div 
                      //+ mse_Phi_lap 
                      //+ mse_U_div
        ;

        // Combine the losses into a Physics Informed Neural Network.
        //mse = mse_data; 
        mse = mse_data + mse_resd; 

        // Optimize weights of the PiNN.
        mse.backward(); 
        optimizer.step();

        std::cout << "Epoch = " << epoch << "\n"
            << "Data MSE = " << mse_data.item<double>() << "\n"
            << "Residual MSE = " << mse_resd.item<double>() << "\n"
            << "Training MSE = " << mse.item<double>() << "\n";
        
        // Write the hiddenLayers_ network structure as a string-formatted python list.
        dataFile << "\"";
        for(decltype(hiddenLayers.size()) i = 0; i < hiddenLayers.size() - 1; ++i)
            dataFile << hiddenLayers[i] << ",";
        dataFile  << hiddenLayers[hiddenLayers.size() - 1] 
            << "\"" << ",";
        // Write the rest of the data. 
        dataFile << optimizerStep << "," << maxIterations << "," 
            << delta_x << "," << epoch << "," 
            << mse_data.item<double>() << "," 
            << mse_resd.item<double>() << ","
            << mse.item<double>() << std::endl;
        
        if (mse.item<double>() < min_mse)
        {
            min_mse = mse.item<double>();
            // Save the "best" model with the minimal MSE over all epochs.
            nn_best = nn;
        }
    }
    
    // Evaluate the best NN. 
    //  - Reinterpret OpenFOAM's output volScalarField as scalar* array 
    volScalarField::pointer Phi_nn_data = Phi_nn.ref().data();
    volVectorField::pointer U_nn_data = U_nn.ref().data();

    //  - Use the scalar* (volScalarField::pointer) to view 
    //    the volScalarField as torch::Tensor without copying data. 
    torch::Tensor Phi_nn_tensor = torch::from_blob(Phi_nn_data, {Phi.size(), 1});
    torch::Tensor U_nn_tensor = torch::from_blob(U_nn_data, {U.size(), 3});
    torch::Tensor PhiU_nn_tensor = torch::cat({Phi_nn_tensor, U_nn_tensor}, 1);

    //  - Evaluate the volumeScalarField vf_nn using the best NN model.
    PhiU_nn_tensor = nn_best->forward(cc_tensor);

    //  - FIXME: 2022-06-01, the C++ PyTorch API does not overwrite the blob object.
    //           If a Model is coded by inheritance, maybe forward(input, output) is
    //           available, that overwrites the data in vf_nn by acting on the 
    //           non-const view of the data given by vf_nn_tensor. TM.
    forAll(Phi_nn, cellI)
    {
        Phi_nn[cellI]  = PhiU_nn_tensor[cellI][0].item<double>();
        U_nn[cellI][0] = PhiU_nn_tensor[cellI][1].item<double>();
        U_nn[cellI][1] = PhiU_nn_tensor[cellI][2].item<double>();
        U_nn[cellI][2] = PhiU_nn_tensor[cellI][3].item<double>();
    }
    //  - Evaluate the vf_nn boundary conditions. 
    Phi_nn.correctBoundaryConditions();
    U_nn.correctBoundaryConditions();

    // Error calculation and output.
    // - Data
    error_Phi == Foam::mag(Phi - Phi_nn);
    scalar error_Phi_l_inf = Foam::max(error_Phi).value();
    scalar error_Phi_mean = Foam::average(error_Phi).value();
    Info << "max(|Phi - Phi|) = " << error_Phi_l_inf << endl; 
    Info << "mean(|Phi - Phi_nn|) = " << error_Phi_mean << endl; 
    error_U == Foam::mag(U - U_nn);
    scalar error_U_l_inf = Foam::max(error_U).value();
    scalar error_U_mean = Foam::average(error_U).value();
    Info << "max(|U - U|) = " << error_U_l_inf << endl; 
    Info << "mean(|U - U_nn|) = " << error_U_mean << endl; 

    // - Gradient  
    volScalarField Phi_nn_lap ("Phi_nn_lap", fvc::laplacian(Phi_nn));
    volScalarField error_lap_Phi ("error_lap_Phi", Phi - Phi_nn);
    volScalarField U_nn_div("U_nn_div", fvc::div(U_nn));
    volScalarField error_div_U ("error_lap_U", mag(U - U_nn));

    // Write fields
    // - Write fields for Phi. Guoliang
    error_Phi.write();
    Phi_nn.write();
    Phi_nn_lap.write();
    Phi_lap.write(); 
    error_lap_Phi.write();
    // -  Write fields for U. Guoliang
    error_U.write();
    U_nn.write();
    U_nn_div.write();
    U_div.write(); 
    error_div_U.write();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
