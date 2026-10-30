#include <cmath>

#include <openturns/DistFunc.hxx>
#include <openturns/EmpiricalBernsteinCopula.hxx>
#include <openturns/Exception.hxx>
#include <openturns/NormalCopulaFactory.hxx>
#include <openturns/SpecFunc.hxx>

#include "otagrum/CorrectedMutualInformation.hxx"
#include "otagrum/Utils.hxx"

namespace OTAGRUM
{

CorrectedMutualInformation::CorrectedMutualInformation(const OT::Sample &data)
  : OT::Object()
{
  if (data.getSize() == 0)
    throw OT::InvalidArgumentException(HERE) << "Error: expected a non-empty sample.";
  data_ = (data.rank() + 1) / (data.getSize() + 2); // Switching data to rank space
  // Split for the Analytic mode, ranks computed on the whole sample then
  // separated, first half for construction, second half for evaluation.
  // OT::Sample::split(index) truncates the sample before index and returns the rest.
  fit_ = data_;
  eval_ = fit_.split(data_.getSize() / 2);
}

void CorrectedMutualInformation::setKMode(KModeTypes kmode)
{
  clearCaches();
  kmode_ = kmode;
}

void CorrectedMutualInformation::setCMode(CModeTypes cmode)
{
  clearCaches();
  cmode_ = cmode;
}

CorrectedMutualInformation::KModeTypes CorrectedMutualInformation::getKMode()
{
  return kmode_;
}

CorrectedMutualInformation::CModeTypes CorrectedMutualInformation::getCMode()
{
  return cmode_;
}

void CorrectedMutualInformation::setAlpha(const double alpha)
{
  alpha_ = alpha;
}

double CorrectedMutualInformation::getAlpha() const
{
  return alpha_;
}

OT::UnsignedInteger CorrectedMutualInformation::getSize() const
{
  return data_.getSize();
}

OT::UnsignedInteger CorrectedMutualInformation::getDimension() const
{
  return data_.getDimension();
}

OT::Description CorrectedMutualInformation::getDataDescription() const
{
  return data_.getDescription();
}

void CorrectedMutualInformation::clearCaches() const
{
  clearHCache();
}

void CorrectedMutualInformation::clearHCache() const
{
  HCache_.clear();
  logPDFCache_.clear();
}

// Get key associated to an OT::Indices in order to store it in cache
// Question : is K mandatory ?
std::string CorrectedMutualInformation::GetKey(OT::Indices l,
    const OT::UnsignedInteger K) const
{
  if (!l.isIncreasing())              // In order to be independent
  {
    std::sort(l.begin(), l.end());  // of the order of indices
  }
  //return l.__str__();
  return l.__str__() + ":" + std::to_string(K);
}

std::string CorrectedMutualInformation::GetKey(OT::Indices l) const
{
  if (!l.isIncreasing())              // In order to be independent
  {
    std::sort(l.begin(), l.end());  // of the order of indices
  }
  return l.__str__();
}

OT::UnsignedInteger CorrectedMutualInformation::sizeForK() const
{
  return kmode_ == KModeTypes::Analytic ? fit_.getSize() : data_.getSize();
}

OT::Point CorrectedMutualInformation::computeLogPDFOnEval(const OT::Indices &variables,
    const int K)
{
  if (variables.getSize() <= 1)
  {
    return OT::Point(eval_.getSize(), 0.0);
  }
  if (cmode_ != CModeTypes::Bernstein)
    throw OT::NotYetImplementedException(HERE) << "Analytic mode requires Bernstein copulas.";
  const std::string key = GetKey(variables, K);
  if (!logPDFCache_.exists(key))
  {
    const OT::EmpiricalBernsteinCopula bc(fit_.getMarginal(variables), K, false);
    const OT::Sample logPDF = bc.computeLogPDF(eval_.getMarginal(variables));
    logPDFCache_.insert(key, logPDF.asPoint());
  }
  return logPDFCache_[key];
}

void CorrectedMutualInformation::BinomialConstants(const OT::UnsignedInteger K,
    double &a, double &S)
{
  // G(r, s) = int_0^1 P(Bin(K-1,u)=r) P(Bin(K-1,u)=s) du
  //         = C(K-1,r) C(K-1,s) Beta(r+s+1, 2K-1-r-s)
  const double n = static_cast<double>(K) - 1.0;
  std::vector< double > lnBinom(K);
  for (OT::UnsignedInteger r = 0; r < K; ++r)
  {
    const double dr = static_cast<double>(r);
    lnBinom[r] = OT::SpecFunc::LogGamma(n + 1.0) - OT::SpecFunc::LogGamma(dr + 1.0)
                 - OT::SpecFunc::LogGamma(n - dr + 1.0);
  }
  a = 0.0;
  S = 0.0;
  for (OT::UnsignedInteger r = 0; r < K; ++r)
  {
    for (OT::UnsignedInteger s = 0; s < K; ++s)
    {
      const double drs = static_cast<double>(r + s);
      const double G = std::exp(lnBinom[r] + lnBinom[s]
                                + OT::SpecFunc::LogBeta(drs + 1.0, 2.0 * n + 1.0 - drs));
      S += G * G;
      if (r == s)
        a += G;
    }
  }
  a *= static_cast<double>(K);
}

double CorrectedMutualInformation::computeAnalyticCorrectedInformation(
  const std::vector< std::pair< OT::Indices, int > > &blocks)
{
  const OT::UnsignedInteger nFit = fit_.getSize();
  const OT::UnsignedInteger nEval = eval_.getSize();
  const OT::UnsignedInteger D = blocks[0].first.getSize();
  const OT::UnsignedInteger K = GetK(nFit, D);

  // Pointwise log-ratio on the evaluation half, its mean and its variance.
  OT::Point logRatio(nEval, 0.0);
  for (const auto &block : blocks)
  {
    const OT::Point logPDF = computeLogPDFOnEval(block.first, K);
    for (OT::UnsignedInteger i = 0; i < nEval; ++i)
      logRatio[i] += block.second * logPDF[i];
  }
  double mean = 0.0;
  for (OT::UnsignedInteger i = 0; i < nEval; ++i)
    mean += logRatio[i];
  mean /= static_cast<double>(nEval);
  double s2 = 0.0;
  if (nEval > 1)
  {
    for (OT::UnsignedInteger i = 0; i < nEval; ++i)
      s2 += (logRatio[i] - mean) * (logRatio[i] - mean);
    s2 /= static_cast<double>(nEval - 1);
  }

  // Centring mu_n: each block of dimension d >= 2 contributes sign * (a^d - 1) / (2 n_fit).
  // Scale sigma_n: the largest block alone, (K^{2D} S^D - 1) / (2 n_fit^2).
  double a = 0.0, S = 0.0;
  BinomialConstants(K, a, S);
  double mu = 0.0;
  for (const auto &block : blocks)
  {
    const OT::UnsignedInteger d = block.first.getSize();
    if (d >= 2)
      mu += block.second * (std::pow(a, static_cast<double>(d)) - 1.0);
  }
  mu = -mu / (2.0 * static_cast<double>(nFit));
  const double dD = static_cast<double>(D);
  const double dK = static_cast<double>(K);
  const double sigma2 = (std::pow(dK, 2.0 * dD) * std::pow(S, dD) - 1.0)
                        / (2.0 * static_cast<double>(nFit) * static_cast<double>(nFit));
  const double sigmaTot = std::sqrt(sigma2 + s2 / static_cast<double>(nEval));
  const double z = OT::DistFunc::qNormal(1.0 - alpha_);

  return mean - (mu + z * sigmaTot);
}

// Compute the (copula) entropy on variables
double CorrectedMutualInformation::computeCrossEntropy(const OT::Indices &variables,
    const int K)
{
  if (kmode_ == KModeTypes::Analytic)
  {
    // Split estimate, minus the mean of the log-density on the evaluation half.
    const OT::Point logPDF = computeLogPDFOnEval(variables, K);
    double H = 0.0;
    for (OT::UnsignedInteger i = 0; i < logPDF.getSize(); ++i)
      H -= logPDF[i];
    return logPDF.getSize() > 0 ? H / static_cast<double>(logPDF.getSize()) : 0.0;
  }
  if (variables.getSize() > 1)    // If there is more than one variable
  {
    //const auto key = GetKey(variables, K);
    std::string key;
    switch (cmode_)
    {
      case CModeTypes::Gaussian:
        key = GetKey(variables);
        break;
      case CModeTypes::Bernstein:
        key = GetKey(variables, K);
        break;
        //default:
    }
    if (!HCache_.exists(key))    // if H(variables) haven't been computed
    {
      auto marginal_data = data_.getMarginal(variables);
      double H = 0.;
      // Is there a way to declare only one copula ?
      // Plus it would cleaner to have no mention of K in the Gaussian case
      auto nc = OT::NormalCopula();
      auto bc = OT::EmpiricalBernsteinCopula();
      switch (cmode_)
      {
        case CModeTypes::Gaussian:
          nc = OT::NormalCopulaFactory().buildAsNormalCopula(marginal_data);
          H = -nc.computeLogPDF(marginal_data).computeMean()[0];
          break;

        case CModeTypes::Bernstein:
          bc = OT::EmpiricalBernsteinCopula(marginal_data, K, false);
          H = -bc.computeLogPDF(marginal_data).computeMean()[0];
          break;

          //default:
          // GUM_ERROR ?

      }
      HCache_.insert(key, H);
      return H;
    }
    else
    {
      return HCache_[key];
    }
  }
  else   // Else entropy is null
  {
    return 0.;
  }
}

OT::UnsignedInteger CorrectedMutualInformation::GetK(const OT::UnsignedInteger size,
    const OT::UnsignedInteger dimension)
{
  return Utils::GetK(size, dimension);
}

double CorrectedMutualInformation::compute2PtInformation(const OT::UnsignedInteger X,
    const OT::UnsignedInteger Y,
    const OT::Indices &U)
{
  OT::UnsignedInteger K = GetK(sizeForK(), U.getSize() + 2);

  double H_XU = computeCrossEntropy(U + X, K);
  double H_YU = computeCrossEntropy(U + Y, K);
  double H_XYU = computeCrossEntropy(U + X + Y, K);
  double H_U = computeCrossEntropy(U, K);

  double IXY_U = H_XU + H_YU - H_XYU - H_U;

  // TO DO ? add a cache for I

  return IXY_U;

}

double CorrectedMutualInformation::compute2PtInformation(const OT::Indices &X,
    const OT::Indices &Y)
{
  OT::UnsignedInteger K = GetK(sizeForK(), X.getSize() + Y.getSize());

  double H_X = computeCrossEntropy(X, K);
  double H_Y = computeCrossEntropy(Y, K);
  double H_XY = computeCrossEntropy(X + Y, K);

  double IXY = H_X + H_Y - H_XY;

  return IXY;
}

double CorrectedMutualInformation::compute2PtPenalty(
  // Commented until a better correction is found
  //const OT::UnsignedInteger X,
  //const OT::UnsignedInteger Y,
  //const OT::Indices &U
)
{
  if (kmode_ == KModeTypes::NoCorr)
  {
    return 0.;
  }
  else if (kmode_ == KModeTypes::Naive)
  {
    return alpha_;
  }
  else
  {
    return 0.;
  }
}

double CorrectedMutualInformation::compute2PtCorrectedInformation(const OT::UnsignedInteger X,
    const OT::UnsignedInteger Y,
    const OT::Indices &U)
{
  if (kmode_ == KModeTypes::Analytic)
  {
    // log-ratio = log c_XYU + log c_U - log c_XU - log c_YU
    return computeAnalyticCorrectedInformation({{U + X + Y, +1}, {U, +1}, {U + X, -1}, {U + Y, -1}});
  }
  // Commented until a better correction is found
  //return compute2PtInformation(X, Y, U) - compute2PtPenalty(X, Y, U);
  return compute2PtInformation(X, Y, U) - compute2PtPenalty();
}

double CorrectedMutualInformation::compute2PtCorrectedInformation(const OT::Indices &X,
    const OT::Indices &Y)
{
  if(X.getSize() == 0 || Y.getSize() == 0)
  {
    return 0.;
  }
  else if (kmode_ == KModeTypes::Analytic)
  {
    // log-ratio = log c_XY - log c_X - log c_Y
    return computeAnalyticCorrectedInformation({{X + Y, +1}, {X, -1}, {Y, -1}});
  }
  else
  {
    return compute2PtInformation(X, Y) - compute2PtPenalty();
  }
}

double CorrectedMutualInformation::compute3PtInformation(const OT::UnsignedInteger X,
    const OT::UnsignedInteger Y,
    const OT::UnsignedInteger Z,
    const OT::Indices &U)
{
  OT::UnsignedInteger K = GetK(sizeForK(), U.getSize() + 3);

  double H_XU = computeCrossEntropy(U + X, K);
  double H_YU = computeCrossEntropy(U + Y, K);
  double H_ZU = computeCrossEntropy(U + Z, K);

  double H_XYU = computeCrossEntropy(U + X + Y, K);
  double H_XZU = computeCrossEntropy(U + X + Z, K);
  double H_YZU = computeCrossEntropy(U + Y + Z, K);

  double H_XYZU = computeCrossEntropy(U + X + Y + Z, K);

  double H_U = computeCrossEntropy(U, K);

  // Maybe it could be better using ICache and the formula
  // I(X;Y;Z|U) = I(X;Y|U) - I(X;Y|U,Z) but what to do about K ?
  //return compute2PtInformation(X, Y, U) - compute2PtInformation(X, Y, U + Z);

  double IXYZ_U = H_XU + H_YU + H_ZU - H_XYU - H_XZU - H_YZU + H_XYZU - H_U;

  return IXYZ_U;
}

double CorrectedMutualInformation::compute3PtPenalty(
  // Commented until a better correction is found
  //const OT::UnsignedInteger X,
  //const OT::UnsignedInteger Y,
  //const OT::UnsignedInteger Z,
  //const OT::Indices &U
)
{
  if (kmode_ == KModeTypes::NoCorr)
  {
    return 0.;
  }
  else if (kmode_ == KModeTypes::Naive)
  {
    return -alpha_;
  }
  else
  {
    return 0.;
  }
}

double CorrectedMutualInformation::compute3PtCorrectedInformation(const OT::UnsignedInteger X,
    const OT::UnsignedInteger Y,
    const OT::UnsignedInteger Z,
    const OT::Indices &U)
{
  // Commented until a better correction is found
  //return compute3PtInformation(X, Y, Z, U) - compute3PtPenalty(X, Y, Z, U);
  return compute3PtInformation(X, Y, Z, U) - compute3PtPenalty();
}

} // namespace OTAGRUM
