// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "edge_se3exp_pointxyz_prior.h"
#include <iostream>

namespace g2o {
  using namespace std;

  EdgeSE3ExpXYZPointPrior::EdgeSE3ExpXYZPointPrior() : BaseUnaryEdge<3, Vector3, VertexSE3Expmap>() {
    information().setIdentity();
  }

  bool EdgeSE3ExpXYZPointPrior::read(std::istream& is) {
    // read measurement
    Vector3 meas;
    for (int i=0; i<3; i++) is >> meas[i];
    setMeasurement(meas);
    // read covariance matrix (upper triangle)
    if (is.good()) {
      for ( int i=0; i<information().rows(); i++)
        for (int j=i; j<information().cols(); j++){
          is >> information()(i,j);
          if (i!=j)
            information()(j,i)=information()(i,j);
        }
    }
    return !is.fail();
  }

  bool EdgeSE3ExpXYZPointPrior::write(std::ostream& os) const {
    for (int i = 0; i<3; i++) os << measurement()[i] << " ";
    for (int i=0; i<information().rows(); i++)
      for (int j=i; j<information().cols(); j++) {
        os << information()(i,j) << " ";
      }
    return os.good();
  }

  void EdgeSE3ExpXYZPointPrior::computeError() {
    const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    const SE3Quat &pt = v->estimate();
    _error = pt.translation() - _measurement;
  }

  void EdgeSE3ExpXYZPointPrior::linearizeOplus(){
      _jacobianOplusXi(0,0) = 0.0;
      _jacobianOplusXi(0,1) = 0.0;
      _jacobianOplusXi(0,2) = 0.0;
      _jacobianOplusXi(0,3) = 1.0;
      _jacobianOplusXi(0,4) = 0.0;
      _jacobianOplusXi(0,5) = 0.0;

      _jacobianOplusXi(1,0) = 0.0;
      _jacobianOplusXi(1,1) = 0.0;
      _jacobianOplusXi(1,2) = 0.0;
      _jacobianOplusXi(1,3) = 0.0;
      _jacobianOplusXi(1,4) = 1.0;
      _jacobianOplusXi(1,5) = 0.0;

      _jacobianOplusXi(2,0) = 0.0;
      _jacobianOplusXi(2,1) = 0.0;
      _jacobianOplusXi(2,2) = 0.0;
      _jacobianOplusXi(2,3) = 0.0;
      _jacobianOplusXi(2,4) = 0.0;
      _jacobianOplusXi(2,5) = 1.0;

  }

  bool EdgeSE3ExpXYZPointPrior::setMeasurementFromState(){
      const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
      _measurement = ((SE3Quat)(v->estimate())).translation();
      return true;
  }
}
