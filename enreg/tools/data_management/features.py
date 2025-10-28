import json
import numba
import torch
import numpy as np


# @numba.njit
def deltaR_etaPhi(eta1, phi1, eta2, phi2):
    """Calculates the angular distance between two objects in the eta-phi coordinates.

    Args:
        eta1 : float
            The eta coordinate of the first object.
        phi1 : float
            The phi coordinate of the first object.
        eta2 : float
            The eta coordinate of the second object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        dR : float
            The angular distance between the two objects
    """
    deta = np.abs(eta1 - eta2)
    dphi = deltaPhi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


# @numba.njit
def deltaR_thetaPhi(theta1, phi1, theta2, phi2):
    """Calculates the angular distance between two objects in the theta-phi coordinates.

    Args:
        theta1 : float
            The theta coordinate of the first object.
        phi1 : float
            The phi coordinate of the first object.
        theta2 : float
            The theta coordinate of the second object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        dR : float
            The angular distance between the two objects
    """
    dtheta = np.abs(theta1 - theta2)
    dphi = deltaPhi(phi1, phi2)
    return np.sqrt(dtheta**2 + dphi**2)


# @numba.njit
def deltaPhi(phi1, phi2):
    """Calculates the difference in azimuthal angle of two objects.

    Args:
        phi1 : float
            The phi coordinate of the first object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        dPhi : float
            The difference in azimuthal angle
    """
    diff = phi1 - phi2
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


# @numba.njit()
def deltaTheta(theta1, theta2):
    """Calculates the difference in polar angle of two objects.

    Args:
        theta1 : float
            The theta coordinate of the first object.
        theta2 : float
            The theta coordinate of the second object.

    Returns:
        dTheta : float
            The difference in polar angle
    """
    return np.abs(theta1 - theta2)


# @numba.njit()
def deltaEta(eta1, eta2):
    """Calculates the difference in pseudorapidity of two objects.

    Args:
        eta1 : float
            The eta coordinate of the first object.
        eta2 : float
            The eta coordinate of the second object.

    Returns:
        dEta : float
            The difference in pseudorapidity
    """
    return np.abs(eta1 - eta2)


# @numba.njit()
def angle3d(theta1, phi1, theta2, phi2):
    """Calculates the 3D angle between two objects given the theta and phi coordinates for them.

    Args:
        theta1 : float
            The theta coordinate of the first object.
        phi1 : float
            The phi coordinate of the first object.
        theta2 : float
            The theta coordinate of the second object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        angle : float
            The 3D angle between the two objects
    """
    u_x1 = np.cos(phi1) * np.sin(theta1)
    u_x2 = np.cos(phi2) * np.sin(theta2)
    u_y1 = np.sin(phi1) * np.sin(theta1)
    u_y2 = np.sin(phi2) * np.sin(theta2)
    u_z1 = np.cos(theta1)
    u_z2 = np.cos(theta2)
    cos_angle = u_x1 * u_x2 + u_y1 * u_y2 + u_z1 * u_z2
    assert cos_angle >= -1.0 and cos_angle <= +1.0
    angle = np.arccos(cos_angle)
    return angle


class FeatureStandardization:
    def __init__(self, method, features, feature_dim, verbosity=-1):
        if verbosity >= 1:
            print("<FeatureStandardization>:")
            print(" method = %s" % method)
            print(" features = %s" % features)
            print(" feature_dim = %i" % feature_dim)

        self.method = method
        self.features = features
        self.feature_dim = feature_dim
        self.verbosity = verbosity

        self.mean = {}
        self.one_over_sigma = {}
        self.dims = {}

        self.is_one_hot_encoded = {}

    def compute_params(self, dataloader):
        if self.verbosity >= 1:
            print("<FeatureStandardization::compute_params>:")

        if self.method == "mean_rms":
            self.compute_params_mean_rms(dataloader)
        elif self.method == "median_quantile":
            self.compute_params_median_quantile(dataloader)
        else:
            raise RuntimeError("Invalid configuration parameter 'method' !!")

        # CV: do not transform one-hot-embedded features
        for feature in self.features:
            for idx in range(len(self.is_one_hot_encoded[feature])):
                if self.is_one_hot_encoded[feature][idx].item():
                    self.mean[feature][idx] = 0.0
                    self.one_over_sigma[feature][idx] = 1.0

        self.reshape_params()

        if self.verbosity >= 1:
            self.print()

    def compute_params_mean_rms(self, dataloader):
        xs = {}
        x2s = {}
        num_particles = 0
        for X, y, weight in dataloader:
            for feature in self.features:
                x = X[feature]

                self.dims[feature] = x.dim()

                if self.feature_dim != (x.dim() - 1):
                    x = torch.swapaxes(x, self.feature_dim, -1)
                x = x.reshape(-1, x.size(dim=-1))
                sum_x = torch.sum(x, dim=0)
                x2 = x * x
                sum_x2 = torch.sum(x2, dim=0)
                if feature not in xs.keys() or feature not in x2s.keys():
                    xs[feature] = sum_x
                    x2s[feature] = sum_x2
                else:
                    xs[feature] = torch.add(xs[feature], sum_x)
                    x2s[feature] = torch.add(x2s[feature], sum_x2)

                if feature not in self.is_one_hot_encoded.keys():
                    self.is_one_hot_encoded[feature] = X[
                        "%s_is_one_hot_encoded" % feature
                    ][0]

            mask = X["mask"]
            num_particles += mask.sum().item()

        # print("num_particles = %i" % num_particles)
        if num_particles == 0:
            raise RuntimeError("Dataset given as function argument is empty !!")

        for feature in self.features:
            x = torch.mul(xs[feature], 1.0 / num_particles)
            x2 = torch.mul(x2s[feature], 1.0 / num_particles)

            mean = x
            var = torch.sub(x2, x * x)
            sigma = torch.sqrt(var)
            torch.clamp(sigma, min=1.0e-8, max=None)
            one_over_sigma = torch.div(torch.tensor(1), sigma)
            if torch.isnan(one_over_sigma).sum().item() > 0.0:
                raise RuntimeError(
                    "Failed to compute standard deviation, because <x^2> - <x>^2 is negative !!"
                )

            self.mean[feature] = mean
            self.one_over_sigma[feature] = one_over_sigma

    def compute_params_median_quantile(self, dataloader):
        feature_values = {}
        for X, y, weight in dataloader:
            for feature in self.features:
                x = X[feature]

                self.dims[feature] = x.dim()

                if self.feature_dim != (x.dim() - 1):
                    x = torch.swapaxes(x, self.feature_dim, -1)
                x = x.reshape(-1, x.size(dim=-1))

                mask = X["mask"]

                if self.feature_dim != (mask.dim() - 1):
                    mask = torch.swapaxes(mask, self.feature_dim, -1)
                mask = mask.reshape(-1, mask.size(dim=-1))

                mask = mask.ge(0.5)

                x_masked = torch.masked_select(x, mask)
                x_masked = x_masked.reshape(-1, x.size(dim=-1))

                if feature not in feature_values.keys():
                    feature_values[feature] = torch.tensor(x_masked)
                else:
                    feature_values[feature] = torch.cat(
                        [feature_values[feature], x_masked]
                    )

                if feature not in self.is_one_hot_encoded.keys():
                    self.is_one_hot_encoded[feature] = X[
                        "%s_is_one_hot_encoded" % feature
                    ][0]

        for feature in self.features:
            median, _ = torch.median(feature_values[feature], dim=0)
            quantile_hi = torch.quantile(feature_values[feature], 0.975, dim=0)
            quantile_lo = torch.quantile(feature_values[feature], 0.025, dim=0)
            width = quantile_hi - quantile_lo

            self.mean[feature] = median
            sigma = 0.5 * width
            torch.clamp(sigma, min=1.0e-8, max=None)
            one_over_sigma = torch.div(torch.tensor(1), sigma)
            self.one_over_sigma[feature] = one_over_sigma

    def reshape_params(self):
        # CV: add dimensions of size one before and after the "feature dimension"
        for feature in self.features:
            for dim in range(self.dims[feature]):
                if dim < self.feature_dim:
                    self.mean[feature] = torch.unsqueeze(self.mean[feature], 0)
                    self.one_over_sigma[feature] = torch.unsqueeze(
                        self.one_over_sigma[feature], 0
                    )
                elif dim > self.feature_dim:
                    self.mean[feature] = torch.unsqueeze(self.mean[feature], -1)
                    self.one_over_sigma[feature] = torch.unsqueeze(
                        self.one_over_sigma[feature], -1
                    )

    def __call__(self, X):
        if self.verbosity >= 4:
            print("<FeatureStandardization::operator()>:")
        X_transformed = {}
        # apply transformation to requested features
        for feature in self.features:
            x = X[feature]
            if self.verbosity >= 4:
                print("before transformation: %s = " % feature, x[0])
            x_transformed = torch.sub(x, self.mean[feature])
            x_transformed = torch.mul(x_transformed, self.one_over_sigma[feature])
            X_transformed[feature] = x_transformed
            if self.verbosity >= 4:
                print(
                    "after transformation: %s = " % feature, X_transformed[feature][0]
                )
        # add features for which no transformation is requested
        for feature in X.keys():
            if feature not in self.features:
                X_transformed[feature] = X[feature]
        return X_transformed

    def load_params(self, filename):
        if self.verbosity >= 1:
            print("<FeatureStandardization::load_params>:")
            print(" filename = %s" % filename)
        file = open(filename, "r")
        cfg = json.load(file)
        file.close()
        if self.verbosity >= 1:
            print("cfg = %s" % cfg)
        for feature in self.features:
            self.mean[feature] = torch.tensor(cfg[feature]["mean"])
            self.one_over_sigma[feature] = torch.tensor(cfg[feature]["one_over_sigma"])
            self.dims[feature] = int(cfg[feature]["dims"])

        self.reshape_params()

        if self.verbosity >= 1:
            self.print()

    def save_params(self, filename):
        if self.verbosity >= 1:
            print("<FeatureStandardization::save_params>:")
            print(" filename = %s" % filename)
        cfg = {}
        for feature in self.features:
            cfg[feature] = {}
            cfg[feature]["mean"] = [
                float(mean)
                for mean in self.mean[feature].squeeze().detach().cpu().numpy()
            ]
            cfg[feature]["one_over_sigma"] = [
                float(one_over_sigma)
                for one_over_sigma in self.one_over_sigma[feature]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            ]
            cfg[feature]["dims"] = self.dims[feature]
        cfg["method"] = self.method
        if self.verbosity >= 1:
            print("cfg = %s" % cfg)
        file = open(filename, "w")
        json.dump(cfg, file)
        file.close()

    def print(self):
        print("<FeatureStandardization::print>:")
        print(" method = %s" % self.method)
        for feature in self.features:
            print("%s:" % feature)
            print(" shape(mean) = ", self.mean[feature].shape)
            print(" mean = ", self.mean[feature].squeeze())
            print(" shape(sigma) = ", self.one_over_sigma[feature].shape)
            print(
                " sigma = ",
                torch.div(torch.tensor(1), self.one_over_sigma[feature].squeeze()),
            )
            print(" dims = ", self.dims[feature])
